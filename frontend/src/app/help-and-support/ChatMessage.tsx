import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  message: string;
  sender: 'user' | 'assistant';
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, sender }) => {
  return (
    <div className={`w-full my-2 ${sender === 'user' ? 'text-right' : 'text-left'}`}>
      <div
        className={`inline-block px-4 py-2 rounded-lg ${
          sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'
        } max-w-xl`}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p className="prose prose-sm max-w-none mb-1">{children}</p>,
            ul: ({ children }) => <ul className="prose prose-sm max-w-none list-disc pl-5">{children}</ul>,
            ol: ({ children }) => <ol className="prose prose-sm max-w-none list-decimal pl-5">{children}</ol>,
            li: ({ children }) => <li className="mb-1">{children}</li>,
            code: ({ children }) => <code className="bg-gray-200 rounded px-1">{children}</code>,
            pre: ({ children }) => <pre className="bg-gray-200 rounded p-2 overflow-x-auto">{children}</pre>,
            a: ({ children, href }) => <a href={href} className="text-blue-600 underline">{children}</a>,
          }}
        >
          {message}
        </ReactMarkdown>
      </div>
    </div>
  );
};

export default ChatMessage; 