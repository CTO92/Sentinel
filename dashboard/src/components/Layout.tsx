import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";

export function Layout() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="ml-60 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-screen-2xl p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
