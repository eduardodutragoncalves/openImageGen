/**
 * One icon system, drawn to the world's rules: a 16-unit grid, 1.5 stroke,
 * butt caps, mitre joins, and corners chamfered at 45° rather than rounded.
 * Nothing here is a glyph or an emoji standing in for a drawing.
 */

type IconProps = { size?: number; className?: string };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "butt" as const,
  strokeLinejoin: "miter" as const,
  "aria-hidden": true,
  focusable: false as const,
});

export function IconPlus({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 3v10M3 8h10" />
    </svg>
  );
}

export function IconMinus({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 8h10" />
    </svg>
  );
}

export function IconClose({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
    </svg>
  );
}

/** The recurring mark: a bare 45° stroke. */
export function IconDiagonal({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 13L13 3" />
    </svg>
  );
}

export function IconSearch({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 3h7v7H3zM10 10l3 3" />
    </svg>
  );
}

export function IconDownload({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 2v8M4.5 7.5L8 11l3.5-3.5M2.5 13.5h11" />
    </svg>
  );
}

export function IconTrash({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M2.5 4h11M6 4V2.5h4V4M4 4l.75 9.5h6.5L12 4" />
    </svg>
  );
}

export function IconUpload({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 11V3M4.5 5.5L8 2l3.5 3.5M2.5 13.5h11" />
    </svg>
  );
}

export function IconCheck({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M3 8.5L6.5 12 13 4.5" />
    </svg>
  );
}

/** Caution: the chamfered plate, not a rounded triangle. */
export function IconCaution({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 2l6 11H2L8 2z" />
      <path d="M8 6.5v3.5M8 11.5v.01" strokeWidth={1.5} />
    </svg>
  );
}

export function IconAlarm({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M5.5 2h5L14 5.5v5L10.5 14h-5L2 10.5v-5L5.5 2z" />
      <path d="M8 5v4M8 11v.01" />
    </svg>
  );
}

export function IconImage({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M2.5 2.5h11v11h-11zM2.5 11l3.5-3.5 3 3 2-2 2.5 2.5" />
    </svg>
  );
}

export function IconLayers({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 2l6 3.5L8 9 2 5.5 8 2zM2 10l6 3.5L14 10" />
    </svg>
  );
}

export function IconChip({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4.5 4.5h7v7h-7zM6.5 1.5v3M9.5 1.5v3M6.5 11.5v3M9.5 11.5v3M1.5 6.5h3M1.5 9.5h3M11.5 6.5h3M11.5 9.5h3" />
    </svg>
  );
}

export function IconKey({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M9.5 2.5h4v4h-4zM9.5 6.5L3 13M5 11l1.5 1.5M3.5 12.5L5 14" />
    </svg>
  );
}

export function IconSun({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M5.5 5.5h5v5h-5zM8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.4 1.4M11.55 11.55l1.4 1.4M12.95 3.05l-1.4 1.4M4.45 11.55l-1.4 1.4" />
    </svg>
  );
}

export function IconMoon({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M13 9.5A5.5 5.5 0 016.5 3 5.5 5.5 0 108 14a5.5 5.5 0 005-4.5z" />
    </svg>
  );
}

export function IconSeed({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M8 2.5L13.5 8 8 13.5 2.5 8 8 2.5z" />
    </svg>
  );
}

export function IconArrowRight({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M2.5 8h11M9.5 4l4 4-4 4" />
    </svg>
  );
}

export function IconRefresh({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M13.5 8a5.5 5.5 0 11-1.9-4.15M13.5 1.5V5h-3.5" />
    </svg>
  );
}
