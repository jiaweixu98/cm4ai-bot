import {
  ArrowLeft,
  ArrowUpRight,
  BookmarkSimple,
  BookOpenText,
  Brain,
  Check,
  ClockCounterClockwise,
  Copy,
  DotsThreeCircle,
  Flag,
  FolderOpen,
  MagnifyingGlass,
  PaperPlaneTilt,
  Paperclip,
  Plus,
  Sparkle,
  Square,
  Stethoscope,
  UsersThree,
  X,
} from "@phosphor-icons/react";

const iconProps = { size: 17, weight: "regular", "aria-hidden": true };

export function IconSend() {
  return <PaperPlaneTilt {...iconProps} weight="fill" />;
}

export function IconStop() {
  return <Square {...iconProps} weight="fill" />;
}

export function IconPlus() {
  return <Plus {...iconProps} />;
}

export function IconHistory() {
  return <ClockCounterClockwise {...iconProps} />;
}

export function IconCopy() {
  return <Copy {...iconProps} />;
}

export function IconRetry() {
  return <ClockCounterClockwise {...iconProps} />;
}

export function IconFlag() {
  return <Flag {...iconProps} />;
}

export function IconClose() {
  return <X {...iconProps} />;
}

export function IconSearch() {
  return <MagnifyingGlass {...iconProps} />;
}

export function IconSpark() {
  return <Sparkle {...iconProps} />;
}

export function IconPaperclip() {
  return <Paperclip {...iconProps} />;
}

export function IconBookmark({ filled = false }) {
  return <BookmarkSimple {...iconProps} weight={filled ? "fill" : "regular"} />;
}

export function IconExternal() {
  return <ArrowUpRight {...iconProps} />;
}

export function IconBack() {
  return <ArrowLeft {...iconProps} />;
}

export function IconCheck() {
  return <Check {...iconProps} weight="bold" />;
}

export function IconMentor() {
  return <BookOpenText {...iconProps} />;
}

export function IconTeam() {
  return <UsersThree {...iconProps} />;
}

const STARTER_ICONS = {
  packaging: FolderOpen,
  governance: DotsThreeCircle,
  grants: Brain,
  multimodal: UsersThree,
  clinical: Stethoscope,
  ehr: BookOpenText,
  mentor: BookOpenText,
  team: UsersThree,
};

export function StarterIcon({ name }) {
  const Icon = STARTER_ICONS[name] || Sparkle;
  return <Icon {...iconProps} />;
}
