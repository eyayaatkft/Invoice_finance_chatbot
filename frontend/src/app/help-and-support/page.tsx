"use client";
import Sidebar from "./Sidebar";
import Chatbot from "./Chatbot";
import * as LucideIcons from "lucide-react";
import { LucideIcon } from "lucide-react";
import React, { useEffect, useState } from "react";

function getLucideIcon(name: string): LucideIcon {
  if (!name) return LucideIcons.HelpCircle;
  // Remove 'lucide:' prefix if present
  let clean = name.replace(/^lucide:/, "");
  // Convert kebab-case to PascalCase
  clean = clean
    .split("-")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join("");
  // @ts-ignore
  return LucideIcons[clean] || LucideIcons.HelpCircle;
}

export default function HelpAndSupportPage() {
  const [themes, setThemes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/help-themes")
      .then((res) => res.json())
      .then((data) => {
        if (data && Array.isArray(data.themes) && data.themes.length > 0) {
          setThemes(data.themes);
        } else {
          setThemes([]);
        }
      })
      .catch(() => setThemes([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen flex bg-gray-50">
      <Sidebar active="Help and Support" />
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12">
        <h1 className="text-4xl font-bold mb-8 text-center">How Can We Help You?</h1>
        {loading ? (
          <div className="text-purple-700 text-lg">Loading themes...</div>
        ) : themes.length > 0 ? (
          <div className="grid gap-8 grid-cols-1 sm:grid-cols-2 md:grid-cols-3 max-w-5xl w-full">
            {themes.map((theme) => {
              const Icon = getLucideIcon(theme.icon);
              return (
                <div key={theme.key || theme.title || theme.label} className="bg-white rounded-xl shadow p-6 flex flex-col items-center hover:shadow-lg transition">
                  <Icon className="w-10 h-10 text-purple-700 mb-4" />
                  <h2 className="text-xl font-semibold mb-2">{theme.title || theme.label}</h2>
                  {(theme.snippet || theme.summary || theme.description) && (
                    <p className="text-gray-600 text-center mt-2">{theme.snippet || theme.summary || theme.description}</p>
                  )}
                </div>
              );
            })}
          </div>
        ) : null}
        <Chatbot />
      </main>
    </div>
  );
} 