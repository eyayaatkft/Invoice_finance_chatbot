import React, { useEffect, useState, useRef } from "react";
import Chatbot from "./Chatbot";
import * as LucideIcons from "lucide-react";
import { LucideIcon } from "lucide-react";
import { setActiveKnowledgeBaseUrl } from "./activeKnowledgeBase";
import { motion, AnimatePresence } from "framer-motion";

export interface HelpSupportWidgetProps {
  url: string; // main site url
}

function getLucideIcon(name: string): LucideIcon {
  if (!name) return LucideIcons.HelpCircle;
  let clean = name.replace(/^lucide:/, "");
  clean = clean
    .split("-")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join("");
  // @ts-ignore
  return LucideIcons[clean] || LucideIcons.HelpCircle;
}

const HelpSupportWidget: React.FC<HelpSupportWidgetProps> = ({ url }) => {
  const [themes, setThemes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [scraped, setScraped] = useState<boolean | null>(null);
  const [error, setError] = useState("");
  const chatbotRef = useRef<any>(null);

  useEffect(() => {
    setActiveKnowledgeBaseUrl(url);
    // Check if KB exists for this URL
    setLoading(true);
    fetch(`/api/knowledge/list`)
      .then(res => res.json())
      .then(data => {
        if (data.urls && data.urls[url]) {
          setScraped(true);
        } else {
          // Not scraped yet, trigger scraping
          setScraped(null); // indicate scraping in progress
          fetch(`/api/knowledge/scrape`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
          })
            .then(res => {
              if (res.ok) {
                setScraped(true);
              } else {
                setScraped(false);
                setError("Failed to scrape knowledge base.");
                setLoading(false);
              }
            })
            .catch(() => {
              setScraped(false);
              setError("Failed to scrape knowledge base.");
              setLoading(false);
            });
        }
      })
      .catch(() => {
        setScraped(false);
        setError("Failed to check knowledge base.");
        setLoading(false);
      });
  }, [url]);

  useEffect(() => {
    if (!url || scraped !== true) return;
    setLoading(true);
    fetch(`/api/help-themes?url=${encodeURIComponent(url)}`)
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
  }, [url, scraped]);

  // Handler to send theme label to Chatbot
  const handleThemeClick = (theme: any) => {
    if (chatbotRef.current && chatbotRef.current.askQuestion) {
      chatbotRef.current.askQuestion(theme.label);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
        <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 w-full">
          <h1 className="text-4xl font-bold mb-8 text-center">How Can We Help You?</h1>
          <div className="text-red-600 text-lg">{error}</div>
        </main>
      </div>
    );
  }

  if (scraped === false) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center">
        <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 w-full">
          <h1 className="text-4xl font-bold mb-8 text-center">How Can We Help You?</h1>
          <div className="text-gray-400 text-lg">No content available.</div>
        </main>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-gray-50 flex flex-col items-center justify-center overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute -top-32 -left-32 w-96 h-96 bg-primary opacity-20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-secondary opacity-20 rounded-full blur-3xl animate-pulse" />
      </div>
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-12 w-full">
        <h1 className="text-4xl font-bold mb-8 text-center text-primary">How Can We Help You?</h1>
        {loading ? (
          <div className="text-primary text-lg">Loading themes...</div>
        ) : themes.length > 0 ? (
          <div className="grid gap-10 grid-cols-1 sm:grid-cols-2 md:grid-cols-2 max-w-5xl w-full">
            <AnimatePresence>
              {themes.map((theme, idx) => {
                const Icon = getLucideIcon(theme.icon);
                return (
                  <motion.div
                    key={theme.key || theme.label}
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 40 }}
                    transition={{ delay: idx * 0.15, duration: 0.7, type: "spring" }}
                    className="bg-white rounded-2xl shadow-xl p-8 flex flex-col items-center hover:shadow-2xl transition-shadow duration-300 relative group cursor-pointer border border-transparent hover:border-secondary overflow-hidden"
                    onClick={() => handleThemeClick(theme)}
                  >
                    {/* Animated background accent */}
                    <motion.div
                      className="absolute inset-0 z-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 0.08 }}
                      transition={{ duration: 0.7 }}
                      style={{ background: `radial-gradient(circle at 70% 30%, var(--color-secondary) 0%, transparent 70%)` }}
                    />
                    <span className="relative z-10 flex items-center justify-center w-16 h-16 rounded-xl mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300" style={{ background: `linear-gradient(135deg, var(--color-primary) 60%, var(--color-secondary) 100%)` }}>
                      <Icon className="w-10 h-10 text-white drop-shadow-lg" />
                    </span>
                    <h2 className="relative z-10 text-2xl font-extrabold mb-2 text-gray-900 group-hover:text-primary transition-colors duration-300 text-center">
                      {theme.label}
                    </h2>
                    {(theme.snippet || theme.summary || theme.description) && (
                      <p className="relative z-10 text-gray-600 text-center mt-2 text-base font-medium group-hover:text-gray-800 transition-colors duration-300">
                        {theme.snippet || theme.summary || theme.description}
                      </p>
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        ) : (
          <div className="text-gray-400 text-lg">No content available.</div>
        )}
        <Chatbot ref={chatbotRef} />
      </main>
    </div>
  );
};

export { HelpSupportWidget }; 