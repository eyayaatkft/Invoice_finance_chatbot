import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const url = req.nextUrl.searchParams.get('url');
  if (!url) return NextResponse.json({ themes: [] }, { status: 400 });
  const res = await fetch(
    process.env.NEXT_PUBLIC_API_BASE_URL + '/help-themes?url=' + encodeURIComponent(url)
  );
  const data = await res.json();
  return NextResponse.json(data);
} 