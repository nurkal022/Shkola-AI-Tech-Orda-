/**
 * Лекция 25: Главная страница чат-приложения
 *
 * "use client" — рендерится в браузере, нужен для useState/useEffect.
 * Без директивы — Server Component (статический рендер на сервере).
 *
 * Архитектура:
 *   page.tsx (собирает компоненты)
 *     ↳ useChat (бизнес-логика)
 *         ↳ useStream (SSE стриминг)
 *     ↳ ChatHistory → ChatMessage
 *     ↳ ChatInput
 */

"use client";

import { useChat } from "@/hooks/useChat";
import ChatHistory from "@/components/ChatHistory";
import ChatInput from "@/components/ChatInput";
import styles from "./page.module.css";

export default function ChatPage() {
  const { session, streamStatus, sendMessage, clearHistory, stopStream } = useChat();

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logo}>⚡</div>
          <div>
            <h1 className={styles.title}>TechOrda AI Chat</h1>
            <p className={styles.subtitle}>Лекция 25 — React + Next.js + Streaming</p>
          </div>
        </div>
        <div className={styles.headerRight}>
          <div
            className={`${styles.statusDot} ${streamStatus === "streaming" ? styles.statusActive : styles.statusIdle}`}
            title={streamStatus === "streaming" ? "Генерирует..." : "Готов"}
          />
          <span className={styles.statusLabel}>
            {streamStatus === "streaming" ? "Генерирует..." : "Готов"}
          </span>
          <button
            className={styles.clearButton}
            onClick={clearHistory}
            disabled={session.messages.length === 0}
          >
            🗑 Очистить
          </button>
        </div>
      </header>

      <div className={styles.body}>
        <ChatHistory messages={session.messages} />
        <ChatInput onSend={sendMessage} status={streamStatus} onStop={stopStream} />
      </div>
    </main>
  );
}
