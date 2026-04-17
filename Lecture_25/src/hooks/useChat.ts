/**
 * Лекция 25: Хук управления историей чата
 *
 * Объединяет:
 * - Хранение истории сообщений (useState)
 * - SSE стриминг (useStream)
 * - Персистентность через localStorage
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import { useStream } from "./useStream";
import type { Message, ChatSession, StreamStatus } from "@/types/chat";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8005";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "sk-secure-pro-001";
const STORAGE_KEY = "techorda_chat_sessions";

function generateId(): string {
  return Math.random().toString(36).slice(2, 11);
}

function createMessage(role: Message["role"], content: string): Message {
  return { id: generateId(), role, content, timestamp: new Date() };
}

interface UseChatReturn {
  session: ChatSession;
  streamStatus: StreamStatus;
  sendMessage: (text: string) => Promise<void>;
  clearHistory: () => void;
  stopStream: () => void;
}

export function useChat(): UseChatReturn {
  const [session, setSession] = useState<ChatSession>(() => ({
    id: generateId(),
    title: "Новый чат",
    messages: [],
    createdAt: new Date(),
    updatedAt: new Date(),
  }));

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as ChatSession;
      saved.messages = saved.messages.map((m) => ({ ...m, timestamp: new Date(m.timestamp) }));
      saved.createdAt = new Date(saved.createdAt);
      saved.updatedAt = new Date(saved.updatedAt);
      setSession(saved);
    } catch {
      // localStorage повреждён — начинаем чисто
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    } catch { /* квота исчерпана */ }
  }, [session]);

  const addMessage = useCallback((msg: Message) => {
    setSession((prev) => ({
      ...prev,
      messages: [...prev.messages, msg],
      updatedAt: new Date(),
    }));
  }, []);

  const updateLastAssistant = useCallback((content: string, done: boolean) => {
    setSession((prev) => {
      const msgs = [...prev.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content, isStreaming: !done };
      }
      return { ...prev, messages: msgs, updatedAt: new Date() };
    });
  }, []);

  const { status: streamStatus, startStream, stop } = useStream({
    onToken: (token) => {
      setSession((prev) => {
        const msgs = [...prev.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role === "assistant") {
          msgs[msgs.length - 1] = { ...last, content: last.content + token, isStreaming: true };
        }
        return { ...prev, messages: msgs };
      });
    },
    onDone: (fullText) => updateLastAssistant(fullText, true),
    onError: (error) => updateLastAssistant(`Ошибка: ${error.message}`, true),
  });

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || streamStatus === "streaming") return;

      addMessage(createMessage("user", text));
      const placeholder = { ...createMessage("assistant", ""), isStreaming: true };
      addMessage(placeholder);

      await startStream(
        `${BACKEND_URL}/v1/chat/stream`,
        { message: text, session_id: session.id, stream: true },
        { Authorization: `Bearer ${API_KEY}` }
      );
    },
    [streamStatus, addMessage, startStream, session.id]
  );

  const clearHistory = useCallback(() => {
    setSession({ id: generateId(), title: "Новый чат", messages: [], createdAt: new Date(), updatedAt: new Date() });
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return { session, streamStatus, sendMessage, clearHistory, stopStream: stop };
}
