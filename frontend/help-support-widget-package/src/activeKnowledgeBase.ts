export function setActiveKnowledgeBaseUrl(url: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('activeKnowledgeBaseUrl', url);
  }
}

export function getActiveKnowledgeBaseUrl(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('activeKnowledgeBaseUrl');
  }
  return null;
} 