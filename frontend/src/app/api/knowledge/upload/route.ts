import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const res = await fetch(process.env.NEXT_PUBLIC_API_BASE_URL + '/upload', {
    method: 'POST',
    body: formData,
  });
  const data = await res.json();
  return NextResponse.json(data);
} 