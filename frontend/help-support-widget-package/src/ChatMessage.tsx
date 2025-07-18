import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion } from 'framer-motion';

interface ChatMessageProps {
  message: string;
  sender: 'user' | 'assistant';
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, sender }) => {
  const isUser = sender === 'user';
  return (
    <div className={`w-full my-3 flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, type: 'spring' }}
        className={`relative px-5 py-3 rounded-2xl max-w-xl shadow-md group transition-all duration-300
          ${isUser
            ? 'bg-gradient-to-tr from-primary to-secondary text-white self-end rounded-br-md hover:shadow-xl'
            : 'bg-white border border-primary text-gray-900 self-start hover:shadow-lg'}
        `}
        style={{
          boxShadow: isUser
            ? '0 4px 16px 0 rgba(3,81,99,0.10)'
            : '0 2px 8px 0 rgba(3,81,99,0.08)'
        }}
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
            a: ({ children, href }) => <a href={href} className="text-secondary underline">{children}</a>,
          }}
        >
          {message}
        </ReactMarkdown>
      </motion.div>
    </div>
  );
};

export default ChatMessage; 