/**
 * Лекция 25: TypeScript типы для чат-приложения
 */

export type Role = "user" | "assistant" | "system";

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  temperature?: number;
  stream?: boolean;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  tokens_used: number;
  latency_ms: number;
}

export type StreamStatus = "idle" | "streaming" | "done" | "error";
