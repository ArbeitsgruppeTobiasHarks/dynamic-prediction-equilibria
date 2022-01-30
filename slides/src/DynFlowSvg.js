import React, { useRef } from 'react';
import TeX from '@matejmazur/react-katex'

const RED = '#a00'
const GREEN = '#0a0'

export const FlowModelSvg = () => {
    const sPos = [25, 100]
    const vPos = [225, 100]
    const tPos = [425, 100]
    return <svg width={500} height={500}>

        <SvgDefs />

        <BaseEdge from={sPos} to={vPos} width={20} />
        <BaseEdge from={vPos} to={tPos} width={10} inEdgeSteps={[
            { start: 0, end: 200, values: [{ color: RED, value: 5 }, { color: GREEN, value: 5 }] }
        ]} queueSteps={[
            { start: -50, end: 0, values: [{ color: RED, value: 5 }, { color: GREEN, value: 5 }] },
            { start: -100, end: -50, values: [{ color: RED, value: 10 }] }
        ]} />
        <Vertex pos={sPos} label={<TeX>s</TeX>} />
        <Vertex pos={vPos} label={<TeX>v</TeX>} />
        <Vertex pos={tPos} label={<TeX>t</TeX>} />
    </svg>
}

export const calcOutflowSteps = (outflow, colors) => {
    const outflowTimes = merge(outflow.map(pwConstant => pwConstant.times))
    // Every two subsequent values in outflowTimes corresponds to a flow step.
    const flowSteps = []
    for (let i = 0; i < outflowTimes.length - 1; i++) {
        // Block from i to i+1
        const start = outflowTimes[i]
        const end = outflowTimes[i + 1]
        const values = []
        for (let c = 0; c < outflow.length; c++) {
            values.push({ color: colors[c], value: outflow[c].eval(start) })
        }
        flowSteps.push({ start, end, values })
    }
    return flowSteps
}

export const splitOutflowSteps = (outflowSteps, queue, transitTime, capacity, t) => {
    const queueSteps = []
    const inEdgeSteps = []

    const queueLength = queue.eval(t) / capacity

    for (let step of outflowSteps) {
        const relStart = step.start - t
        const relEnd = step.end - t

        const queueStart = Math.max(transitTime - relEnd, -queueLength)
        const queueEnd = Math.min(transitTime - relStart, 0)
        if (queueStart < queueEnd) {
            queueSteps.push({ start: queueStart, end: queueEnd, values: step.values })
        }

        const inEdgeStart = Math.max(relStart, 0)
        const inEdgeEnd = Math.min(relEnd, transitTime)
        if (inEdgeStart < inEdgeEnd) {
            inEdgeSteps.push({ start: inEdgeStart, end: inEdgeEnd, values: step.values })
        }
    }

    return { queueSteps, inEdgeSteps }
}



const merge = (lists) => {
    const indices = lists.map(() => 0)
    let curVal = Math.min(...lists.map(list => list[0]))
    const merged = [curVal]
    while (true) {
        let min = Infinity
        let listIndex = -1
        for (let i = 0; i < lists.length; i++) {
            if (indices[i] < lists[i].length - 1 && lists[i][indices[i] + 1] <= min) {
                min = lists[i][indices[i] + 1]
                listIndex = i
            }
        }

        if (listIndex == -1) {
            return merged
        } else {
            merged.push(min)
            indices[listIndex] += 1
        }
    }
}

export const SvgDefs = ({ svgIdPrefix }) => (<>
    <linearGradient id={`${svgIdPrefix}fade-grad`} x1="0" y1="1" y2="0" x2="0">
        <stop offset="0" stopColor='white' stopOpacity="0.5" />
        <stop offset="1" stopColor='white' stopOpacity="0.2" />
    </linearGradient>
    <mask id={`${svgIdPrefix}fade-mask`} maskContentUnits="objectBoundingBox">
        <rect width="1" height="1" fill={`url(#${svgIdPrefix}fade-grad)`} />
    </mask>
</>
)

export const BaseEdge = ({ svgIdPrefix, visible, from, to, width = 10, inEdgeSteps = [], queueSteps = [] }) => {
    const padding = 40
    const arrowHeadWidth = 10
    const delta = [to[0] - from[0], to[1] - from[1]]
    const norm = Math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    // start = from + (to - from)/|to - from| * 30
    const pad = [delta[0] / norm * padding, delta[1] / norm * padding]
    const edgeStart = [from[0] + pad[0], from[1] + pad[1]]
    const deg = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI
    //return <path d={`M${start[0]},${start[1]}L${end[0]},${end[1]}`} />
    const scaledNorm = norm - 2 * padding - arrowHeadWidth
    const scale = scaledNorm / norm

    return <g transform={`rotate(${deg}, ${edgeStart[0]}, ${edgeStart[1]})`} style={{ transition: "opacity 0.2s" }} opacity={visible ? 1 : 0}>
        <path stroke="black" fill="lightgray" d={d.M(edgeStart[0] + scaledNorm, edgeStart[1] - width) + d.l(arrowHeadWidth, width) + d.l(-arrowHeadWidth, width) + d.z} />
        <rect
            x={edgeStart[0]} y={edgeStart[1] - width / 2}
            width={scaledNorm} height={width} fill="white" stroke="none"
        />
        {
            inEdgeSteps.map(({ start, end, values }) => {
                const s = values.reduce((acc, { value }) => acc + value, 0)
                let y = edgeStart[1] - s / 2
                return values.map(({ color, value }) => {
                    const myY = y
                    y += value
                    return <rect fill={color} x={edgeStart[0] + scaledNorm - scale * end} y={myY} width={(end - start) * scale} height={value} />
                })
            }).flat()
        }
        <g mask={`url(#${svgIdPrefix}fade-mask)`}>
            {
                queueSteps.map(({ start, end, values }) => {
                    let x = edgeStart[0] - width
                    return values.slice(0).reverse().map(({ color, value }) => {
                        const myX = x
                        x += value
                        return <rect fill={color} x={myX} y={edgeStart[1] - width + start * scale} width={value} height={(end - start) * scale} />
                    })
                }).flat()
            }
        </g>
        <path stroke="gray" style={{ transition: "opacity 0.2s" }} opacity={queueSteps.length > 0 ? 1 : 0}
            fill="none" d={d.M(...edgeStart) + d.c(-width / 2, 0, -width / 2, 0, -width / 2, -width)} />
        <rect
            x={edgeStart[0]} y={edgeStart[1] - width / 2}
            width={scaledNorm} height={width} stroke="black" fill="none"
        />
    </g>
}


export const FlowEdge = ({ svgIdPrefix, from, to, outflowSteps, queue, t, capacity, transitTime, visible = true }) => {
    const { inEdgeSteps, queueSteps } = splitOutflowSteps(outflowSteps, queue, transitTime, capacity, t)

    return <BaseEdge svgIdPrefix={svgIdPrefix} visible={visible} from={from} to={to} width={capacity} inEdgeSteps={inEdgeSteps} queueSteps={queueSteps} />
}


export const Vertex = ({ label, pos, visible = true }) => {
    const radius = 20
    const [cx, cy] = pos
    return <g style={{ transition: "opacity 0.2s" }} opacity={visible ? 1 : 0}>
        <circle cx={cx} cy={cy} r={radius} stroke="black" fill="white" />
        {label ? (<ForeignObjectLabel cx={cx} cy={cy}>{label}</ForeignObjectLabel>) : null}
    </g>
}

export const ForeignObjectLabel = ({ cx, cy, width = 40, height = 40, children }) => (
    <foreignObject x={cx - width / 2} y={cy - height / 2} width={width} height={height}>
        <div style={{ width, height, display: 'grid', justifyContent: 'center', alignItems: 'center' }}>
            {children}
        </div>
    </foreignObject>
)



export const StopWatch = ({ t, y, x, size }) => {
    const cx = x + size / 2
    const cy = y + size / 2
    return <>
        <path d={d.M(cx, y) + d.v(-5) + d.h(-3) + d.h(6)} fill="none" stroke="black" strokeWidth={size / 20} />
        <path transform={`rotate(30 ${cx} ${cy})`} d={d.M(cx, y) + d.v(-3) + d.h(-2) + d.h(4)} fill="none" stroke="black" strokeWidth={size / 20} />
        <circle cx={cx} cy={cy} r={size / 2} stroke="black" fill="white" />
        <line transform={`rotate(${0} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${30} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${60} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${90} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${120} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${150} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${180} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${210} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${240} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${270} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${300} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <line transform={`rotate(${330} ${cx} ${cy})`} x1={cx} x2={cx} y1={y} y2={y + size / 10} stroke="gray" />
        <circle cx={cx} cy={cy} r={size / 15} fill="black" />
        <line transform={`rotate(${t / 200 * 360} ${cx} ${cy})`} x1={cx} x2={cx} y1={cy} y2={y + size / 6} stroke="black" />
    </>
}

export const d = {
    M: (x, y) => `M${x} ${y}`,
    c: (dx1, dy1, dx2, dy2, x, y) => `c${dx1} ${dy1} ${dx2} ${dy2} ${x} ${y}`,
    l: (x, y) => `l${x} ${y}`,
    L: (x, y) => `L${x} ${y}`,
    h: (x) => `h${x}`,
    H: (x) => `H${x}`,
    v: (y) => `v${y}`,
    m: (x, y) => `m${x} ${y}`,
    z: 'z'
}
