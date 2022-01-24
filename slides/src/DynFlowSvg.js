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
        <BaseEdge from={vPos} to={tPos} width={10} flowSteps={[
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

export const calcFlowSteps = (flow, edge, colors) => {
    const outflow = flow.outflow[edge.id]
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

export const splitFlowSteps = (outflowSteps, queue, travelTime, t) => {
    const queueSteps = []
    const inEdgeSteps = []

    const queueLength = queue.eval(t)

    for (let step of outflowSteps) {
        const relStart = step.start - t
        const relEnd = step.end - t

        const queueStart = Math.max(relStart - travelTime, -queueLength)
        const queueEnd = Math.min(relEnd - travelTime, 0)
        if (queueStart < queueEnd) {
            queueSteps.push({ start: queueStart, end: queueEnd, values: step.values })
        }

        const inEdgeStart = Math.max(relStart, 0)
        const inEdgeEnd = Math.min(relEnd, travelTime)
        if (inEdgeStart < inEdgeEnd) {
            inEdgeSteps.push({ start: inEdgeStart, end: inEdgeEnd, values: step.values })
        }
    }

    return { queueSteps, inEdgeSteps }
}



const merge = (lists) => {
    const merged = [0]
    let curVal = 0
    const indices = lists.map(() => 0)
    while (true) {
        let min = curVal
        let listIndex = -1
        for (let i = 0; i < lists.length; i++) {
            if (indices[i] < lists[i].length - 1 && lists[i][indices[i] + 1] < min) {
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

const SvgDefs = () => (<>
    <linearGradient id="fade-grad" x1="0" y1="1" y2="0" x2="0">
        <stop offset="0" stopColor='white' stopOpacity="0.5" />
        <stop offset="1" stopColor='white' stopOpacity="0.2" />
    </linearGradient>
    <mask id="fade-mask" maskContentUnits="objectBoundingBox">
        <rect width="1" height="1" fill="url(#fade-grad)" />
    </mask>
</>
)

const BaseEdge = ({ from, to, width = 10, flowSteps = [], queueSteps = [] }) => {
    const padding = 40
    const delta = [to[0] - from[0], to[1] - from[1]]
    const norm = Math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    // start = from + (to - from)/|to - from| * 30
    const pad = [delta[0] / norm * padding, delta[1] / norm * padding]
    const edgeStart = [from[0] + pad[0], from[1] + pad[1]]
    const deg = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI
    //return <path d={`M${start[0]},${start[1]}L${end[0]},${end[1]}`} />
    const scaledNorm = norm - 2 * padding - width
    const scale = scaledNorm / norm

    return <g transform={`rotate(${deg}, ${edgeStart[0]}, ${edgeStart[1]})`}>
        <path stroke="black" fill="lightgray" d={d.M(edgeStart[0] + scaledNorm, edgeStart[1] - width) + d.l(width, width) + d.l(-width, width) + d.z} />
        <rect
            x={edgeStart[0]} y={edgeStart[1] - width / 2}
            width={scaledNorm} height={width} fill="white" stroke="none"
        />
        {
            flowSteps.map(({ start, end, values }) => {
                const s = values.reduce((acc, { value }) => acc + value, 0)
                let y = edgeStart[1] - s / 2
                return values.map(({ color, value }) => {
                    const myY = y
                    y += value
                    return <rect fill={color} x={edgeStart[0] + scaledNorm - scale * end} y={myY} width={(end - start) * scale} height={value} />
                })
            }).flat()
        }
        <g mask="url(#fade-mask)">
            {
                queueSteps.map(({ start, end, values }) => {
                    const s = values.reduce((acc, { value }) => acc + value, 0)
                    let x = edgeStart[0] - width
                    return values.map(({ color, value }) => {
                        const myX = x
                        x += value
                        return <rect fill={color} x={myX} y={edgeStart[1] - width + start * scale} width={value} height={(end - start) * scale} />
                    })
                }).flat()
            }
        </g>
        {queueSteps.length > 0 ? <path stroke="gray" fill="none" d={d.M(...edgeStart) + d.c(-width / 2, 0, -width / 2, 0, -width / 2, -width)} /> : null}
        <rect
            x={edgeStart[0]} y={edgeStart[1] - width / 2}
            width={scaledNorm} height={width} stroke="black" fill="none"
        />
    </g>
}

const Vertex = ({ label, pos }) => {
    const radius = 20
    const [cx, cy] = pos
    return <>
        <circle cx={cx} cy={cy} r={radius} stroke="black" fill="white" />
        {label ? (<foreignObject x={cx - radius} y={cy - radius} width={2 * radius} height={2 * radius}>
            <div style={{ width: 2 * radius, height: 2 * radius, display: 'grid', justifyContent: 'center', alignItems: 'center' }}>
                {label}
            </div></foreignObject>) : null}
    </>
}

const d = {
    M: (x, y) => `M${x} ${y}`,
    c: (dx1, dy1, dx2, dy2, x, y) => `c${dx1} ${dy1} ${dx2} ${dy2} ${x} ${y}`,
    l: (x, y) => `l${x} ${y}`,
    z: 'z'
}
