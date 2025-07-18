import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { MessageCircle } from "lucide-react";
import ChatMessage from "./ChatMessage";
import { getActiveKnowledgeBaseUrl } from "./activeKnowledgeBase";
import { motion, AnimatePresence } from "framer-motion";

interface ChatTurn {
  user: string;
  assistant: string;
}

const Chatbot = forwardRef((props, ref) => {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [currentUrl, setCurrentUrl] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setCurrentUrl(getActiveKnowledgeBaseUrl());
    }
  }, [open]);

  useEffect(() => {
    if (open) {
      const activeUrl = getActiveKnowledgeBaseUrl();
      if (!activeUrl) return;
      fetch(`/api/chat/history?url=${encodeURIComponent(activeUrl)}`)
        .then((res) => res.json())
        .then((data) => setHistory(data || []));
    }
  }, [open]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, open]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    const activeUrl = getActiveKnowledgeBaseUrl();
    if (!input.trim()) return;
    if (!activeUrl || typeof activeUrl !== 'string' || !activeUrl.startsWith('http')) {
      setError("No active knowledge base selected. Please set one in Knowledge Management.");
      return;
    }
    setLoading(true);
    setError("");
    const userMsg = input;
    setHistory((h) => [...h, { user: userMsg, assistant: "..." }]);
    setInput("");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg, url: activeUrl }),
      });
      if (!res.ok) {
        setError("Backend error: " + (await res.text()));
        setHistory((h) => h.slice(0, -1));
        return;
      }
      const data = await res.json();
      setHistory((h) => {
        const newHist = [...h];
        newHist[newHist.length - 1] = { user: userMsg, assistant: data.answer };
        return newHist;
      });
    } catch (err) {
      setError("Something went wrong. Please try again.");
      setHistory((h) => h.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  useImperativeHandle(ref, () => ({
    askQuestion: (question: string) => {
      if (!question.trim()) return;
      setHistory((h) => [...h, { user: question, assistant: "..." }]);
      setOpen(true);
      (async () => {
        setLoading(true);
        setError("");
        try {
          const activeUrl = getActiveKnowledgeBaseUrl();
          const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, url: activeUrl }),
          });
          if (!res.ok) {
            setError("Backend error: " + (await res.text()));
            setHistory((h) => h.slice(0, -1));
            return;
          }
          const data = await res.json();
          setHistory((h) => {
            const newHist = [...h];
            newHist[newHist.length - 1] = { user: question, assistant: data.answer };
            return newHist;
          });
        } catch (err) {
          setError("Something went wrong. Please try again.");
          setHistory((h) => h.slice(0, -1));
        } finally {
          setLoading(false);
        }
      })();
    }
  }));

  return (
    <>
      {/* Floating Button */}
      {!open && (
        <button
          className="fixed bottom-8 right-8 z-50 bg-gradient-to-tr from-primary to-secondary text-white rounded-full shadow-xl w-16 h-16 flex items-center justify-center text-3xl hover:scale-110 hover:shadow-2xl transition-all duration-300"
          onClick={() => setOpen((o) => !o)}
          aria-label="Open chat"
        >
          <MessageCircle className="w-8 h-8" />
        </button>
      )}
      {/* Floating Chat Modal */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 80 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 80 }}
            transition={{ duration: 0.4, type: "spring" }}
            className="fixed bottom-0 right-0 z-50 w-full sm:w-96 max-w-full bg-white rounded-t-2xl sm:rounded-2xl shadow-xl flex flex-col border border-primary/20 overflow-hidden m-0 pb-0"
            style={{ boxShadow: "0 8px 32px 0 rgba(3,81,99,0.15)", maxHeight: '90vh', height: '48rem' }}
          >
            {/* Decorative blurred background */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute -top-16 -left-16 w-40 h-40 bg-primary opacity-20 rounded-full blur-2xl animate-pulse" />
              <div className="absolute -bottom-16 -right-16 w-40 h-40 bg-secondary opacity-20 rounded-full blur-2xl animate-pulse" />
            </div>
            {/* Header always on top */}
            <div className="flex items-center justify-between px-6 py-4 border-b bg-gradient-to-tr from-primary to-secondary rounded-t-2xl sm:rounded-t-2xl sticky top-0 z-10">
              <span className="text-white font-bold text-lg tracking-wide">Chat Assistant</span>
              <button onClick={() => setOpen(false)} className="text-white text-xl hover:bg-primary/30 rounded-full w-8 h-8 flex items-center justify-center transition">
                ✖
              </button>
            </div>
            {/* Scrollable chat messages area */}
            <div className="flex-1 min-h-0 overflow-y-auto px-4 py-3 pt-15 space-y-8 bg-gradient-to-br from-white via-primary/5 to-white">
              {history.length === 0 && <div className="text-gray-400 text-center">No conversation yet.</div>}
              {history.map((turn, idx) => (
                <React.Fragment key={idx}>
                  <ChatMessage message={turn.user} sender="user" />
                  <ChatMessage message={turn.assistant} sender="assistant" />
                </React.Fragment>
              ))}
              <div ref={chatEndRef} />
            </div>
            {error && <div className="px-6 pb-2 text-red-600 text-xs">{error}</div>}
            {/* Fixed input area at the bottom */}
            <form
              onSubmit={sendMessage}
              className="flex items-center px-3 py-2"
            >
              <input
                type="text"
                className="flex-1 border border-primary rounded-full px-4 py-2 mr-2 focus:outline-none focus:ring-2 focus:ring-secondary text-sm bg-primary/5 placeholder-gray-400 transition"
                placeholder="Type your question..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={loading}
              />
              <button
                type="submit"
                className="bg-gradient-to-tr from-primary to-secondary text-white px-5 py-2 rounded-full font-semibold shadow hover:from-primary hover:to-secondary transition disabled:opacity-50 text-sm"
                disabled={loading}
              >
                {loading ? (
                  <span className="animate-spin inline-block mr-2">⏳</span>
                ) : (
                  "Send"
                )}
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
});

export default Chatbot; 