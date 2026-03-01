import type { OpenClawConfig } from "../config/config.js";
import {
  addChannelAllowFromStoreEntry,
  approveChannelPairingCode,
  listChannelPairingRequests,
  readChannelAllowFromStore,
  upsertChannelPairingRequest,
} from "../pairing/pairing-store.js";

const PROVIDER = "dingtalk" as const;

export type DingTalkPairingListEntry = {
  userId: string;
  code: string;
  createdAt: string;
  lastSeenAt: string;
  name?: string;
};

export async function readDingTalkAllowFromStore(params: {
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<string[]> {
  return readChannelAllowFromStore(
    PROVIDER,
    params.env ?? process.env,
    params.accountId ?? "default",
  );
}

export async function addDingTalkAllowFromStoreEntry(params: {
  entry: string;
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<{ changed: boolean; allowFrom: string[] }> {
  return addChannelAllowFromStoreEntry({
    channel: PROVIDER,
    entry: params.entry,
    accountId: params.accountId,
    env: params.env,
  });
}

export async function listDingTalkPairingRequests(params: {
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<DingTalkPairingListEntry[]> {
  const list = await listChannelPairingRequests(
    PROVIDER,
    params.env ?? process.env,
    params.accountId,
  );
  return list.map((r) => ({
    userId: r.id,
    code: r.code,
    createdAt: r.createdAt,
    lastSeenAt: r.lastSeenAt,
    name: r.meta?.name,
  }));
}

export async function upsertDingTalkPairingRequest(params: {
  userId: string;
  name?: string;
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<{ code: string; created: boolean }> {
  return upsertChannelPairingRequest({
    channel: PROVIDER,
    id: params.userId,
    accountId: params.accountId ?? "default",
    env: params.env,
    meta: { name: params.name },
  });
}

export async function approveDingTalkPairingCode(params: {
  code: string;
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<{ userId: string; entry?: DingTalkPairingListEntry } | null> {
  const res = await approveChannelPairingCode({
    channel: PROVIDER,
    code: params.code,
    accountId: params.accountId,
    env: params.env,
  });
  if (!res) {
    return null;
  }
  const entry = res.entry
    ? {
        userId: res.entry.id,
        code: res.entry.code,
        createdAt: res.entry.createdAt,
        lastSeenAt: res.entry.lastSeenAt,
        name: res.entry.meta?.name,
      }
    : undefined;
  return { userId: res.id, entry };
}

export async function resolveDingTalkEffectiveAllowFrom(params: {
  cfg: OpenClawConfig;
  accountId?: string;
  env?: NodeJS.ProcessEnv;
}): Promise<{ dm: string[]; group: string[] }> {
  const env = params.env ?? process.env;
  const dtCfg = params.cfg.channels?.dingtalk;
  const accountCfg = params.accountId ? dtCfg?.accounts?.[params.accountId] : undefined;
  const allowFrom = accountCfg?.allowFrom ?? dtCfg?.allowFrom ?? [];
  const groupAllowFrom = accountCfg?.groupAllowFrom ?? dtCfg?.groupAllowFrom ?? [];

  const cfgAllowFrom = allowFrom
    .map((v) => String(v).trim())
    .filter(Boolean)
    .map((v) => v.replace(/^dingtalk:/i, ""))
    .filter((v) => v !== "*");
  const cfgGroupAllowFrom = groupAllowFrom
    .map((v) => String(v).trim())
    .filter(Boolean)
    .map((v) => v.replace(/^dingtalk:/i, ""))
    .filter((v) => v !== "*");
  const storeAllowFrom = await readDingTalkAllowFromStore({
    accountId: params.accountId,
    env,
  });

  const dm = Array.from(new Set([...cfgAllowFrom, ...storeAllowFrom]));
  const group = Array.from(
    new Set([
      ...(cfgGroupAllowFrom.length > 0 ? cfgGroupAllowFrom : cfgAllowFrom),
      ...storeAllowFrom,
    ]),
  );
  return { dm, group };
}
