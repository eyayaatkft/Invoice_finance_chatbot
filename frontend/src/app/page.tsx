import { redirect } from 'next/navigation';

export default function Home() {
  redirect('/help-and-support');
  return null;
}
