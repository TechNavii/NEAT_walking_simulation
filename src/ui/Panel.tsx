import type { PropsWithChildren, ReactNode } from 'react'

export function Panel(
  props: PropsWithChildren<{ title?: string; className?: string; right?: ReactNode }>,
) {
  return (
    <div className={`panel ${props.className ?? ''}`.trim()}>
      {(props.title || props.right) && (
        <div className="panelHeader">
          <div className="panelTitle">{props.title}</div>
          <div className="panelRight">{props.right}</div>
        </div>
      )}
      <div className="panelBody">{props.children}</div>
    </div>
  )
}
