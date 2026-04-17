/**
 * Лекция 25: Хук для SSE-стриминга
 *
 * Server-Sent Events (SSE) — однонаправленный поток сервер → клиент.
 * Каждый чанк: data: {"token": "..."}\n\n
 * Конец потока: data: [DONE]
 */

"use client";

import { useState, useCallback, useRef } from "react";
import type { StreamStatus } from "@/types/chat";

interface UseStreamOptions {
  onToken?: (token: string) => void;
  onDone?: (fullText: string) => void;
  onError?: (error: Error) => void;
}

interface UseStreamReturn {
  status: StreamStatus;
  text: string;
  startStream: (url: string, body: object, headers?: HeadersInit) => Promise<void>;
  stop: () => void;
}

export function useStream(options: UseStreamOptions = {}): UseStreamReturn {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [text, setText] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setStatus("done");
  }, []);

  const startStream = useCallback(
    async (url: string, body: object, headers: HeadersInit = {}) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus("streaming");
      setText("");
      let accumulated = "";

      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...headers },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: "Ошибка сервера" }));
          throw new Error(err.detail ?? `HTTP ${response.status}`);
        }

        if (!response.body) throw new Error("Сервер не поддерживает стриминг");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") break;

            try {
              const parsed = JSON.parse(data) as { token?: string; text?: string };
              const token = parsed.token ?? parsed.text ?? "";
              if (token) {
                accumulated += token;
                setText(accumulated);
                options.onToken?.(token);
              }
            } catch {
              // keepalive или комментарий — пропускаем
            }
          }
        }

        setStatus("done");
        options.onDone?.(accumulated);
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          setStatus("done");
          return;
        }
        const err = error instanceof Error ? error : new Error("Неизвестная ошибка");
        setStatus("error");
        options.onError?.(err);
      }
    },
    [options]
  );

  return { status, text, startStream, stop };
}
