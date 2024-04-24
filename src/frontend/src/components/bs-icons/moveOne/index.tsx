import React, { forwardRef } from "react";
import { ReactComponent as MoveOne } from "./MoveOne.svg";

export const MoveOneIcon = forwardRef<
    SVGSVGElement & { className: any },
    React.PropsWithChildren<{ className?: string }>
>(({ className, ...props }, ref) => {
    const _className = 'transition text-gray-950 ' + (className || '')
    return <MoveOne ref={ref} {...props} className={_className} />;
});
