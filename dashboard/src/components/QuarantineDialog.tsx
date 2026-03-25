import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { X, ShieldAlert, ShieldCheck, Search, Skull } from "lucide-react";
import { cn } from "@/lib/utils";
import { HealthBadge } from "./HealthBadge";
import { QuarantineAction, type QuarantineDirective } from "@/api/types";

interface QuarantineDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  gpuUuid: string;
  gpuHostname: string;
  currentState: string;
  onConfirm: (directive: QuarantineDirective) => void;
  isSubmitting?: boolean;
}

const actionConfig: Record<
  QuarantineAction,
  { label: string; icon: React.ReactNode; description: string; color: string }
> = {
  [QuarantineAction.QUARANTINE]: {
    label: "Quarantine",
    icon: <ShieldAlert className="h-4 w-4" />,
    description:
      "Remove this GPU from the scheduling pool. It will no longer receive workloads until reinstated.",
    color: "text-health-quarantined",
  },
  [QuarantineAction.REINSTATE]: {
    label: "Reinstate",
    icon: <ShieldCheck className="h-4 w-4" />,
    description:
      "Return this GPU to the active scheduling pool. Ensure confidence in GPU health before reinstating.",
    color: "text-health-healthy",
  },
  [QuarantineAction.DEEP_TEST]: {
    label: "Schedule Deep Test",
    icon: <Search className="h-4 w-4" />,
    description:
      "Run extended diagnostic probes to thoroughly evaluate GPU health. This may take several hours.",
    color: "text-health-deep-test",
  },
  [QuarantineAction.CONDEMN]: {
    label: "Condemn",
    icon: <Skull className="h-4 w-4" />,
    description:
      "Permanently mark this GPU as failed. It will require hardware replacement before returning to service.",
    color: "text-health-condemned",
  },
};

export function QuarantineDialog({
  open,
  onOpenChange,
  gpuUuid,
  gpuHostname,
  currentState,
  onConfirm,
  isSubmitting = false,
}: QuarantineDialogProps) {
  const [selectedAction, setSelectedAction] = useState<QuarantineAction>(
    QuarantineAction.QUARANTINE,
  );
  const [reason, setReason] = useState("");

  const handleConfirm = () => {
    onConfirm({
      gpu_uuid: gpuUuid,
      action: selectedAction,
      reason,
      initiated_by: "dashboard_user",
      evidence_ids: [],
    });
  };

  const config = actionConfig[selectedAction];

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-6 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <Dialog.Title className="text-lg font-semibold">
              GPU Action
            </Dialog.Title>
            <Dialog.Close className="rounded-sm opacity-70 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>

          <div className="mb-4 rounded-md bg-muted p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-mono">{gpuUuid}</p>
                <p className="text-xs text-muted-foreground">{gpuHostname}</p>
              </div>
              <HealthBadge state={currentState} />
            </div>
          </div>

          <div className="mb-4">
            <label className="text-sm font-medium mb-2 block">Action</label>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(actionConfig).map(([action, cfg]) => (
                <button
                  key={action}
                  onClick={() => setSelectedAction(action as QuarantineAction)}
                  className={cn(
                    "flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors",
                    selectedAction === action
                      ? "border-primary bg-primary/10"
                      : "border-border hover:bg-muted",
                  )}
                >
                  <span className={cfg.color}>{cfg.icon}</span>
                  {cfg.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-4 rounded-md bg-muted/50 p-3 text-sm text-muted-foreground">
            {config.description}
          </div>

          <div className="mb-6">
            <label
              htmlFor="quarantine-reason"
              className="text-sm font-medium mb-2 block"
            >
              Reason
            </label>
            <textarea
              id="quarantine-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Provide a reason for this action..."
              rows={3}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none"
            />
          </div>

          <div className="flex justify-end gap-3">
            <Dialog.Close className="rounded-md border border-border px-4 py-2 text-sm hover:bg-muted transition-colors">
              Cancel
            </Dialog.Close>
            <button
              onClick={handleConfirm}
              disabled={!reason.trim() || isSubmitting}
              className={cn(
                "rounded-md px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
                selectedAction === QuarantineAction.CONDEMN
                  ? "bg-health-condemned text-white hover:bg-health-condemned/90"
                  : selectedAction === QuarantineAction.REINSTATE
                    ? "bg-health-healthy text-white hover:bg-health-healthy/90"
                    : "bg-primary text-primary-foreground hover:bg-primary/90",
              )}
            >
              {isSubmitting ? "Submitting..." : `Confirm ${config.label}`}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
