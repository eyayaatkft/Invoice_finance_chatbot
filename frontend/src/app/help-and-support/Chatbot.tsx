"use client";
import { MessageCircle } from "lucide-react";
import React, { useState, useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";

interface ChatTurn {
  user: string;
  assistant: string;
}

const Chatbot = () => {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load chat history when opened
  useEffect(() => {
    if (open) {
      fetch("/api/chat/history")
        .then((res) => res.json())
        .then((data) => setHistory(data || []));
    }
  }, [open]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, open]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    const userMsg = input;
    setHistory((h) => [...h, { user: userMsg, assistant: "..." }]);
    setInput("");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg }),
      });
      const data = await res.json();
      setHistory((h) => {
        const newHist = [...h];
        // Replace the last assistant placeholder with the real answer
        newHist[newHist.length - 1] = { user: userMsg, assistant: data.answer };
        return newHist;
      });
    } catch {
      setError("Something went wrong. Please try again.");
      setHistory((h) => h.slice(0, -1)); // Remove the placeholder
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        className="fixed bottom-8 right-8 z-50 bg-purple-700 text-white rounded-full shadow-lg w-16 h-16 flex items-center justify-center text-3xl hover:bg-purple-800 transition"
        onClick={() => setOpen((o) => !o)}
        aria-label="Open chat"
      >
        <MessageCircle className="w-8 h-8" />
      </button>
      {/* Floating Chat Modal */}
      {open && (
        <div className="fixed bottom-28 right-8 z-50 w-96 max-w-full bg-white rounded-xl shadow-2xl flex flex-col border border-purple-200">
          <div className="flex items-center justify-between px-4 py-3 border-b bg-purple-700 rounded-t-xl">
            <span className="text-white font-semibold">Chat Assistant</span>
            <button onClick={() => setOpen(false)} className="text-white text-lg">✖</button>
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3" style={{ maxHeight: '350px' }}>
            {history.length === 0 && <div className="text-gray-400 text-center">No conversation yet.</div>}
            {history.map((turn, idx) => (
              <React.Fragment key={idx}>
                <ChatMessage message={turn.user} sender="user" />
                <ChatMessage message={turn.assistant} sender="assistant" />
              </React.Fragment>
            ))}
            <div ref={chatEndRef} />
          </div>
          {error && <div className="px-4 pb-2 text-red-600 text-xs">{error}</div>}
          <form
            onSubmit={sendMessage}
            className="flex items-center px-4 py-3 border-t bg-white rounded-b-xl"
          >
            <input
              type="text"
              className="flex-1 border rounded-full px-4 py-2 mr-2 focus:outline-none focus:ring-2 focus:ring-purple-400 text-sm"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />
            <button
              type="submit"
              className="bg-purple-700 text-white px-4 py-2 rounded-full font-semibold hover:bg-purple-800 transition disabled:opacity-50 text-sm"
              disabled={loading}
            >
              {loading ? (
                <span className="animate-spin inline-block mr-2">⏳</span>
              ) : (
                "Send"
              )}
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default Chatbot; 