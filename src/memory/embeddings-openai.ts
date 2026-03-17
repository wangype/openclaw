import type { SsrFPolicy } from "../infra/net/ssrf.js";
import type { EmbeddingInput } from "./embedding-inputs.js";
import { sanitizeAndNormalizeEmbedding } from "./embedding-vectors.js";
import { normalizeEmbeddingModelWithPrefixes } from "./embeddings-model-normalize.js";
import { fetchRemoteEmbeddingVectors } from "./embeddings-remote-fetch.js";
import {
  createRemoteEmbeddingProvider,
  resolveRemoteEmbeddingClient,
} from "./embeddings-remote-provider.js";
import type { EmbeddingProvider, EmbeddingProviderOptions } from "./embeddings.js";
import { withRemoteHttpResponse } from "./remote-http.js";

export type OpenAiEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
  /** Set when the provider is routed through a DashScope-compatible endpoint. */
  dashscope?: {
    outputDimensionality?: number;
  };
};

export const DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small";
const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const OPENAI_MAX_INPUT_TOKENS: Record<string, number> = {
  "text-embedding-3-small": 8192,
  "text-embedding-3-large": 8192,
  "text-embedding-ada-002": 8191,
};

// --- DashScope multimodal embedding support ---

const DASHSCOPE_MULTIMODAL_EMBEDDING_BASE_URL =
  "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding";

// Models served via DashScope's multimodal-embedding API (not the standard /v1/embeddings path).
export const OPENAI_DASHSCOPE_MULTIMODAL_MODELS = new Set([
  "multimodal-embedding-v1",
  "qwen3-vl-embedding",
]);

export const OPENAI_DASHSCOPE_MAX_INPUT_TOKENS: Record<string, number> = {
  "multimodal-embedding-v1": 8192,
  "qwen3-vl-embedding": 8192,
};

type DashScopeContentItem = {
  text?: string;
  image?: string;
  audio?: string;
  video?: string;
};

type DashScopeEmbeddingEntry = {
  index: number;
  embedding: number[];
  type: "text" | "image" | "audio" | "video";
};

type DashScopeEmbeddingResponse = {
  output?: { embeddings?: DashScopeEmbeddingEntry[] };
  request_id?: string;
};

/**
 * Convert an EmbeddingInput into a DashScope contents array.
 * Inline-data parts are sent as data URLs ("data:<mimeType>;base64,<data>").
 */
export function buildOpenAiDashScopeContents(input: EmbeddingInput): DashScopeContentItem[] {
  if (!input.parts?.length) {
    return [{ text: input.text }];
  }
  const item: DashScopeContentItem = {};
  for (const part of input.parts) {
    if (part.type === "text") {
      item.text = part.text;
    } else {
      const dataUrl = `data:${part.mimeType};base64,${part.data}`;
      const [mediaType] = part.mimeType.split("/");
      if (mediaType === "image") {
        item.image = dataUrl;
      } else if (mediaType === "audio") {
        item.audio = dataUrl;
      } else if (mediaType === "video") {
        item.video = dataUrl;
      } else if (!item.text) {
        item.text = input.text;
      }
    }
  }
  return [item];
}

/**
 * Pick the best single vector from a DashScope per-modality response.
 * Preference: image > audio > video > text.
 */
export function extractOpenAiDashScopeEmbedding(entries: DashScopeEmbeddingEntry[]): number[] {
  if (entries.length === 0) {
    return [];
  }
  const preference: Array<DashScopeEmbeddingEntry["type"]> = ["image", "audio", "video", "text"];
  for (const type of preference) {
    const found = entries.find((e) => e.type === type);
    if (found) {
      return sanitizeAndNormalizeEmbedding(found.embedding);
    }
  }
  return sanitizeAndNormalizeEmbedding(entries[0].embedding);
}

async function fetchDashScopeEmbedding(
  client: OpenAiEmbeddingClient,
  input: EmbeddingInput,
): Promise<number[]> {
  const endpoint = `${client.baseUrl.replace(/\/$/, "")}/multimodal-embedding`;
  const body = {
    model: client.model,
    input: { contents: buildOpenAiDashScopeContents(input) },
    parameters: {
      output_type: "dense" as const,
      ...(client.dashscope?.outputDimensionality != null
        ? { dimension: client.dashscope.outputDimensionality }
        : undefined),
    },
  };

  return await withRemoteHttpResponse({
    url: endpoint,
    ssrfPolicy: client.ssrfPolicy,
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json", ...client.headers },
      body: JSON.stringify(body),
    },
    onResponse: async (res) => {
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`openai dashscope embeddings failed: ${res.status} ${text}`);
      }
      const payload = (await res.json()) as DashScopeEmbeddingResponse;
      return extractOpenAiDashScopeEmbedding(payload.output?.embeddings ?? []);
    },
  });
}

// --- Model normalization ---

export function normalizeOpenAiModel(model: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_OPENAI_EMBEDDING_MODEL,
    prefixes: ["openai/"],
  });
}

/**
 * Returns true when the model should be routed through DashScope's
 * multimodal-embedding API instead of the standard /v1/embeddings endpoint.
 */
export function isOpenAiDashScopeMultimodalModel(model: string): boolean {
  return OPENAI_DASHSCOPE_MULTIMODAL_MODELS.has(model);
}

// --- Provider factory ---

export async function createOpenAiEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<{ provider: EmbeddingProvider; client: OpenAiEmbeddingClient }> {
  const client = await resolveOpenAiEmbeddingClient(options);

  if (isOpenAiDashScopeMultimodalModel(client.model)) {
    // DashScope multimodal path: custom request/response format, embedBatchInputs supported.
    const embedSingle = async (input: EmbeddingInput): Promise<number[]> =>
      await fetchDashScopeEmbedding(client, input);

    const embedBatchInputs = async (inputs: EmbeddingInput[]): Promise<number[][]> => {
      if (inputs.length === 0) {
        return [];
      }
      // DashScope has no batch endpoint; issue requests sequentially.
      const results: number[][] = [];
      for (const input of inputs) {
        results.push(await embedSingle(input));
      }
      return results;
    };

    const embedBatch = async (texts: string[]): Promise<number[][]> =>
      await embedBatchInputs(texts.map((text) => ({ text })));

    return {
      provider: {
        id: "openai",
        model: client.model,
        maxInputTokens: OPENAI_DASHSCOPE_MAX_INPUT_TOKENS[client.model],
        embedQuery: async (text) => {
          if (!text.trim()) {
            return [];
          }
          return await embedSingle({ text });
        },
        embedBatch,
        embedBatchInputs,
      },
      client,
    };
  }

  // Standard OpenAI-compatible path.
  const url = `${client.baseUrl.replace(/\/$/, "")}/embeddings`;
  const embedBatchInputs = async (inputs: EmbeddingInput[]): Promise<number[][]> => {
    if (inputs.length === 0) {
      return [];
    }
    return await fetchRemoteEmbeddingVectors({
      url,
      headers: client.headers,
      ssrfPolicy: client.ssrfPolicy,
      body: { model: client.model, input: inputs.map((i) => i.text) },
      errorPrefix: "openai embeddings failed",
    });
  };

  const baseProvider = createRemoteEmbeddingProvider({
    id: "openai",
    client,
    errorPrefix: "openai embeddings failed",
    maxInputTokens: OPENAI_MAX_INPUT_TOKENS[client.model],
  });

  return {
    provider: { ...baseProvider, embedBatchInputs },
    client,
  };
}

export async function resolveOpenAiEmbeddingClient(
  options: EmbeddingProviderOptions,
): Promise<OpenAiEmbeddingClient> {
  const model = normalizeOpenAiModel(options.model);
  const isDashScope = isOpenAiDashScopeMultimodalModel(model);

  // For DashScope models, default baseUrl points to the DashScope endpoint.
  const defaultBaseUrl = isDashScope
    ? DASHSCOPE_MULTIMODAL_EMBEDDING_BASE_URL
    : DEFAULT_OPENAI_BASE_URL;

  const base = await resolveRemoteEmbeddingClient({
    provider: "openai",
    options,
    defaultBaseUrl,
    normalizeModel: normalizeOpenAiModel,
  });

  if (isDashScope) {
    return {
      ...base,
      dashscope: {
        outputDimensionality:
          typeof options.outputDimensionality === "number" && options.outputDimensionality > 0
            ? options.outputDimensionality
            : undefined,
      },
    };
  }

  return base;
}
