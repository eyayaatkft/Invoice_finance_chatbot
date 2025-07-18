import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const url = req.nextUrl.searchParams.get('url');
  const backendUrl = process.env.NEXT_PUBLIC_API_BASE_URL + '/chat/history' + (url ? `?url=${encodeURIComponent(url)}` : '');
  const res = await fetch(backendUrl);
  const data = await res.json();
  return NextResponse.json(data);
} 