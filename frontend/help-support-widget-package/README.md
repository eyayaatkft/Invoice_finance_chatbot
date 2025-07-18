# Help Support Widget

A reusable React Help & Support Widget with chatbot functionality, styled with Tailwind CSS.

## Installation

```bash
npm install help-support-widget
```

## Peer Dependencies
You must have these installed in your project:
- react >=18
- react-dom >=18
- framer-motion >=7
- lucide-react >=0.525.0
- react-markdown >=10.1.0
- remark-gfm >=4.0.1
- tailwindcss >=3

## Usage

```tsx
import { HelpSupportWidget } from 'help-support-widget';

function App() {
  return <HelpSupportWidget url="https://your-site.com" />;
}
```

## Tailwind & CSS Setup
- Ensure Tailwind CSS is set up in your project.
- Add the following to your global CSS or Tailwind config:
  - Custom colors: `--color-primary`, `--color-secondary`
  - Classes: `bg-primary`, `bg-secondary`, `text-primary`, `text-secondary`
- Example (in `globals.css`):

```css
:root {
  --color-primary: #0ea5e9;
  --color-secondary: #f59e42;
}
.bg-primary { background-color: var(--color-primary); }
.bg-secondary { background-color: var(--color-secondary); }
.text-primary { color: var(--color-primary); }
.text-secondary { color: var(--color-secondary); }
```

## API Requirements
The widget expects the following API endpoints to exist in your backend:
- `/api/knowledge/list`
- `/api/knowledge/scrape`
- `/api/help-themes`
- `/api/chat`
- `/api/chat/history`

Alternatively, you can fork and adapt the widget to use your own endpoints.

## Development
- Build: `npm run build`
- Entry: `src/index.ts`

---
MIT License 