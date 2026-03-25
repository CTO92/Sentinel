import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Cpu,
  ScrollText,
  ShieldAlert,
  GitGraph,
  Settings,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Fleet Overview" },
  { to: "/gpu", icon: Cpu, label: "GPU Detail" },
  { to: "/audit", icon: ScrollText, label: "Audit Trail" },
  { to: "/quarantine", icon: ShieldAlert, label: "Quarantine Queue" },
  { to: "/trust", icon: GitGraph, label: "Trust Graph" },
];

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-60 flex-col border-r border-border bg-card">
      {/* Logo */}
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
        <Shield className="h-6 w-6 text-primary" />
        <span className="text-lg font-bold tracking-tight">SENTINEL</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-3 overflow-y-auto scrollbar-thin">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-border p-3">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )
          }
        >
          <Settings className="h-4 w-4" />
          Settings
        </NavLink>
        <div className="mt-3 px-3 text-[10px] text-muted-foreground">
          SENTINEL v0.1.0
        </div>
      </div>
    </aside>
  );
}
