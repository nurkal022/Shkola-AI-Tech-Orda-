/**
 * Лекция 25: Список сообщений с автоскроллом
 */

"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/types/chat";
import ChatMessage from "./ChatMessage";
import styles from "./ChatHistory.module.css";

export default function ChatHistory({ messages }: { messages: Message[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>💬</div>
        <h2 className={styles.emptyTitle}>Начните диалог</h2>
        <p className={styles.emptyText}>
          Введите вопрос ниже. Ответы генерируются в режиме реального времени.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.list}>
      {messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)}
      <div ref={bottomRef} aria-hidden="true" />
    </div>
  );
}
