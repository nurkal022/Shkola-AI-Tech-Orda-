/**
 * Лекция 25: Компонент одного сообщения
 * Принцип Single Responsibility: только отображает одно сообщение.
 */

import type { Message } from "@/types/chat";
import styles from "./ChatMessage.module.css";

function formatTime(date: Date): string {
  return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`${styles.wrapper} ${isUser ? styles.userWrapper : styles.assistantWrapper}`}>
      <div className={`${styles.avatar} ${isUser ? styles.userAvatar : styles.aiAvatar}`}>
        {isUser ? "Вы" : "AI"}
      </div>
      <div className={styles.bubble}>
        <p className={styles.text}>
          {message.content}
          {message.isStreaming && <span className={styles.cursor} aria-hidden="true" />}
        </p>
        <span className={styles.time}>{formatTime(message.timestamp)}</span>
      </div>
    </div>
  );
}
