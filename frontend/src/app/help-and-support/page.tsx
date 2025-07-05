"use client";
import { Search } from "lucide-react";
import Sidebar from "./Sidebar";
import dynamic from "next/dynamic";
import React from "react";

const Chatbot = dynamic(() => import("./Chatbot"), { ssr: false });

export default function HelpAndSupportPage() {
  return (
    <div className="min-h-screen flex bg-gray-50">
      <Sidebar active="Help and Support" />
      <main className="flex-1 flex flex-col px-0 md:px-8 py-0 md:py-8">
        <div className="max-w-6xl w-full mx-auto rounded-2xl bg-white shadow-sm p-0 md:p-10 min-h-screen">
          <div className="flex flex-col items-center pt-10">
            <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-8 text-center">
              How Can We Help You?
            </h1>
            {/* Search Bar */}
            <div className="relative w-full max-w-xl mb-12">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="What question or problem..."
                className="w-full pl-12 pr-4 py-4 border border-gray-200 rounded-lg shadow-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none text-lg"
              />
            </div>
            {/* Abstract Illustration */}
            <div className="mb-12 flex items-center justify-center w-full">
              <div className="relative w-[400px] h-[120px] mx-auto">
                {/* Top row */}
                <div className="absolute top-0 left-12 w-16 h-8 bg-teal-100 rounded-full" />
                <div className="absolute top-0 left-48 w-24 h-10 bg-yellow-100 rounded-full" />
                {/* Circles */}
                <div className="absolute top-8 left-0 w-16 h-16 bg-teal-400 rounded-full" />
                <div className="absolute top-8 left-24 w-16 h-16 bg-blue-400 rounded-full" />
                <div className="absolute top-8 left-48 w-16 h-16 bg-purple-400 rounded-full" />
                <div className="absolute top-8 left-72 w-16 h-16 bg-green-400 rounded-full" />
                {/* Bottom squares */}
                <div className="absolute top-24 left-2 w-10 h-6 bg-orange-400 rounded" />
                <div className="absolute top-24 left-28 w-10 h-6 bg-yellow-400 rounded" />
                <div className="absolute top-24 left-52 w-10 h-6 bg-red-400 rounded" />
                <div className="absolute top-24 left-76 w-10 h-6 bg-teal-400 rounded" />
                {/* Lower right oval */}
                <div className="absolute top-32 left-64 w-16 h-8 bg-orange-100 rounded-full" />
              </div>
            </div>
            {/* Two-column layout */}
            <div className="grid md:grid-cols-2 gap-8 w-full max-w-4xl mx-auto mt-8">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Popular Questions</h2>
                <div className="text-gray-700 font-medium mb-1">Top questions</div>
                <p className="text-gray-500 text-sm leading-relaxed">
                  Lorem ipsum, dolor sit amet consectetur adipisicing elit. Nam quae nemo, exercitationem totam pariatur dolorum possimus eaque adipisci inventore praesentium! Porro, tempore eum nesciunt corporis ipsum vitae culpa veniam? Ipsa.
                </p>
              </div>
              <div>
                <h2 className="text-2xl font-bold text-orange-500 mb-2">Ask questions</h2>
                <p className="text-gray-500 text-sm leading-relaxed mb-6">
                  Lorem ipsum dolor sit amet consectetur. Gravida tincidunt dignissim aliquam accumsan sed malesuada leo. Etiam magna eu consectetur viverra et scelerisque.
                </p>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-orange-500">Contact Support</h3>
                  <div className="space-y-2 text-sm">
                    <div>
                      <span className="text-gray-600">Call to our Help center</span>
                      <div className="text-orange-500">+977</div>
                    </div>
                    <div>
                      <span className="text-gray-600">Email us</span>
                      <div className="text-orange-500">Support@Qena.com</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* Floating Chatbot Button */}
        <Chatbot />
      </main>
    </div>
  );
} 