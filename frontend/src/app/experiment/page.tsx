"use client";

import React from "react";
import { HelpSupportWidget } from "../../help-support-widget/HelpSupportWidget";
import { useSearchParams } from "next/navigation";

export default function ExperimentPage() {
  const searchParams = useSearchParams();
  const demoUrl = searchParams.get("url");

  return (
    <div className="min-h-screen  flex flex-col items-center justify-center">
      <div className="w-full max-w-full">
        {demoUrl ? (
          <HelpSupportWidget url={demoUrl} />
        ) : (
          <div className="text-gray-500">Please provide a <code>?url=...</code> query parameter in the address bar.</div>
        )}
      </div>
    </div>
  );
} 