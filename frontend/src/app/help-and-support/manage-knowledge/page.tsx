"use client";
import React, { useEffect, useState } from "react";
import Sidebar from "../Sidebar";

interface KnowledgeTracking {
  files: Record<string, string>;
  urls: Record<string, string>;
}

export default function ManageKnowledgePage() {
  const [knowledge, setKnowledge] = useState<KnowledgeTracking>({ files: {}, urls: {} });
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [message, setMessage] = useState("");

  // Fetch tracked files and URLs
  const fetchKnowledge = async () => {
    setLoading(true);
    const res = await fetch("/api/knowledge/list");
    const data = await res.json();
    setKnowledge(data);
    setLoading(false);
  };

  useEffect(() => {
    fetchKnowledge();
  }, []);

  // Handlers for upload and scrape
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setMessage("");
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/knowledge/upload", { method: "POST", body: formData });
    const data = await res.json();
    setMessage(data.success ? "File uploaded and embedded!" : data.error || "Upload failed");
    setFile(null);
    await fetchKnowledge();
  };
  const handleScrape = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;
    setMessage("");
    const res = await fetch("/api/knowledge/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    setMessage(data.success ? "URL scraped and embedded!" : data.error || "Scrape failed");
    setUrl("");
    await fetchKnowledge();
  };

  // Re-embed and remove handlers
  const handleReembed = async (item: string, type: "file" | "url") => {
    setMessage("");
    setLoading(true);
    await fetch("/api/knowledge/reembed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item, type }),
    });
    setMessage("Re-embedded!");
    await fetchKnowledge();
    setLoading(false);
  };
  const handleRemove = async (item: string, type: "file" | "url") => {
    setMessage("");
    setLoading(true);
    await fetch("/api/knowledge/remove", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item, type }),
    });
    setMessage("Removed!");
    await fetchKnowledge();
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex bg-gray-50">
      <Sidebar active="Knowledge Management" />
      <main className="flex-1 flex flex-col items-center justify-center px-8 py-12">
        <div className="max-w-4xl w-full">
          <h1 className="text-4xl font-bold text-gray-900 mb-8 text-center">Knowledge Management</h1>
          <div className="mb-6 flex flex-col md:flex-row gap-6">
            {/* Upload File */}
            <form onSubmit={handleUpload} className="flex flex-col gap-2 border p-4 rounded-lg w-full md:w-1/2 bg-white">
              <label className="font-semibold">Upload File</label>
              <input type="file" onChange={handleFileChange} />
              <button type="submit" className="bg-purple-700 text-white px-4 py-2 rounded mt-2">Upload</button>
            </form>
            {/* Scrape URL */}
            <form onSubmit={handleScrape} className="flex flex-col gap-2 border p-4 rounded-lg w-full md:w-1/2 bg-white">
              <label className="font-semibold">Scrape URL</label>
              <input type="url" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://example.com" className="px-2 py-1 border rounded" />
              <button type="submit" className="bg-purple-700 text-white px-4 py-2 rounded mt-2">Scrape</button>
            </form>
          </div>
          {message && <div className="mb-4 text-green-700 font-semibold text-center">{message}</div>}
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-2">Tracked Files</h2>
            {Object.keys(knowledge.files).length === 0 ? <div className="text-gray-500">No files tracked.</div> : (
              <ul className="space-y-2">
                {Object.entries(knowledge.files).map(([file, mtime]) => (
                  <li key={file} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                    <span className="truncate" title={file}>{file}</span>
                    <span className="text-xs text-gray-400 ml-2">{mtime}</span>
                    <button onClick={() => handleReembed(file, "file")} className="ml-2 text-blue-600 hover:underline" disabled={loading}>Re-embed</button>
                    <button onClick={() => handleRemove(file, "file")} className="ml-2 text-red-600 hover:underline" disabled={loading}>Remove</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <h2 className="text-lg font-semibold mb-2">Tracked URLs</h2>
            {Object.keys(knowledge.urls).length === 0 ? <div className="text-gray-500">No URLs tracked.</div> : (
              <ul className="space-y-2">
                {Object.entries(knowledge.urls).map(([url, ts]) => (
                  <li key={url} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                    <span className="truncate" title={url}>{url}</span>
                    <span className="text-xs text-gray-400 ml-2">{ts}</span>
                    <button onClick={() => handleReembed(url, "url")} className="ml-2 text-blue-600 hover:underline" disabled={loading}>Re-scrape</button>
                    <button onClick={() => handleRemove(url, "url")} className="ml-2 text-red-600 hover:underline" disabled={loading}>Remove</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {loading && <div className="mt-4 text-purple-700 text-center">Loading...</div>}
        </div>
      </main>
    </div>
  );
} 