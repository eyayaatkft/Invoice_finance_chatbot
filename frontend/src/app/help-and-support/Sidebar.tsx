"use client";
import React, { useState } from "react";
import {
  Home,
  FileText,
  Users,
  CreditCard,
  HelpCircle,
  Settings,
  User,
  BookOpen
} from "lucide-react";
import Link from "next/link";

const menuItems = [
  { icon: Home, label: "Overview", href: "/overview" },
  { icon: FileText, label: "Invoice", href: "/invoice" },
  { icon: Users, label: "Agreements", href: "/agreements" },
  { icon: CreditCard, label: "Disbursements", href: "/disbursements" },
  { icon: HelpCircle, label: "Help and Support", href: "/help-and-support" },
  { icon: BookOpen, label: "Knowledge Management", href: "/help-and-support/manage-knowledge" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

const Sidebar = ({ active }: { active?: string }) => {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <>
      {/* Mobile Toggle */}
      <button
        className="md:hidden fixed top-4 left-4 z-50 bg-purple-900 text-white p-2 rounded-lg shadow-lg"
        onClick={() => setCollapsed((c) => !c)}
        aria-label="Toggle sidebar"
      >
        {collapsed ? <span>&#10005;</span> : <span>&#9776;</span>}
      </button>
      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 h-full min-h-screen z-40 bg-purple-900 text-white flex flex-col transition-transform duration-300
          w-64 border-r border-purple-800
          ${collapsed ? "-translate-x-full" : "translate-x-0"}
          md:static md:translate-x-0 md:w-64
        `}
      >
        {/* Logo */}
        <div className="p-6 border-b border-purple-800">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
              <div className="w-4 h-4 bg-purple-900 rounded"></div>
            </div>
            <span className="text-xl font-bold">qena</span>
            <span className="text-xs bg-purple-800 px-2 py-1 rounded">INVOICE</span>
          </div>
        </div>
        {/* Navigation */}
        <nav className="flex-1 py-6 overflow-y-auto">
          <ul className="space-y-2 px-4">
            {menuItems.map((item) => {
              const isActive = active === item.label;
              return (
                <li key={item.label}>
                  <Link
                    href={item.href}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                      isActive
                        ? "bg-purple-800 text-white"
                        : "text-purple-200 hover:bg-purple-800 hover:text-white"
                    }`}
                    onClick={() => setCollapsed(true)}
                  >
                    <item.icon className="h-5 w-5" />
                    <span>{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
        {/* User Profile */}
        <div className="p-4 border-t border-purple-800 mt-auto">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-700 rounded-full flex items-center justify-center">
              <User className="h-5 w-5" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium">Abraham Drage</div>
              <div className="text-xs text-purple-300 truncate">rebel859566@editbit.c...</div>
            </div>
          </div>
        </div>
      </aside>
      {/* Overlay for mobile */}
      {collapsed && (
        <div
          className="fixed inset-0 z-30 bg-black bg-opacity-30 md:hidden"
          onClick={() => setCollapsed(false)}
        />
      )}
    </>
  );
};

export default Sidebar; 