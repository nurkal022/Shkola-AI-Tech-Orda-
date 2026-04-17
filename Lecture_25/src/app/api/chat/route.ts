/**
 * Лекция 25: Next.js Route Handler — прокси к FastAPI (Lecture 22)
 * Файл: src/app/api/chat/route.ts → URL: POST /api/chat
 */

import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8005";
const API_KEY = process.env.BACKEND_API_KEY ?? "sk-secure-pro-001";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const isStream = body.stream === true;

  const backendRes = await fetch(`${BACKEND}/v1/chat${isStream ? "/stream" : ""}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify(body),
  });

  if (!backendRes.ok) {
    const err = await backendRes.text();
    return new Response(err, { status: backendRes.status });
  }

  if (isStream) {
    return new Response(backendRes.body, {
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
    });
  }

  return Response.json(await backendRes.json());
}
