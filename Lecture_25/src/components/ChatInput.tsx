/**
 * Лекция 25: Компонент ввода сообщения
 * - Управляемый textarea (controlled component)
 * - Enter = отправить, Shift+Enter = новая строка
 * - Автовысота textarea
 */

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type { StreamStatus } from "@/types/chat";
import styles from "./ChatInput.module.css";

interface ChatInputProps {
  onSend: (text: string) => void;
  status: StreamStatus;
  onStop: () => void;
}

export default function ChatInput({ onSend, status, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isStreaming = status === "streaming";

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue("");
  }, [value, isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className={styles.container}>
      <div className={styles.inputRow}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Введите сообщение… (Enter — отправить, Shift+Enter — перенос)"
          rows={1}
          disabled={isStreaming}
        />
        {isStreaming ? (
          <button className={`${styles.button} ${styles.stopButton}`} onClick={onStop}>■ Стоп</button>
        ) : (
          <button className={`${styles.button} ${styles.sendButton}`} onClick={handleSend} disabled={!value.trim()}>▶ Send</button>
        )}
      </div>
      <p className={styles.hint}>
        {isStreaming ? "Генерирую ответ…" : "Enter — отправить · Shift+Enter — новая строка"}
      </p>
    </div>
  );
}
