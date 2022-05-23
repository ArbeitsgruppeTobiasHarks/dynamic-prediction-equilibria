import React from 'react';
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

export const calcOutflowSteps = (outflow, commodities) => {
    const outflowTimes = merge(outflow.map(pwConstant => pwConstant.times))
    // Every two subsequent values in outflowTimes corresponds to a flow step.
    const flowSteps = []
    for (let i = 0; i < outflowTimes.length - 1; i++) {
        // Block from i to i+1
        const start = outflowTimes[i]
        const end = outflowTimes[i + 1]
        const values = []
        for (let c = 0; c < outflow.length; c++) {
            values.push({ color: commodities[c].color, value: outflow[c].eval(start) })
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

export const BaseEdge = ({ multiGroup, translate, svgIdPrefix, waitingTimeScale, transitTime, visible, from, to, offset, strokeWidth, flowScale, capacity, inEdgeSteps = [], queueSteps = [] }) => {
    const width = flowScale * capacity
    const padding = offset
    const arrowHeadWidth = offset / 2
    const arrowHeadHeight = multiGroup ? width : 2 * width
    const delta = [to[0] - from[0], to[1] - from[1]]
    const norm = Math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    // start = from + (to - from)/|to - from| * 30
    const pad = [delta[0] / norm * padding, delta[1] / norm * padding]
    const edgeStart = [from[0] + pad[0], from[1] + pad[1]]
    const deg = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI
    //return <path d={`M${start[0]},${start[1]}L${end[0]},${end[1]}`} />
    const normOffsetted = norm - 2 * padding - arrowHeadWidth
    const scale = normOffsetted / norm
    const scaledTranslate = -translate * flowScale

    return <g transform={`rotate(${deg}, ${edgeStart[0]}, ${edgeStart[1]}) translate(0 ${scaledTranslate})`} style={{ transition: "opacity 0.2s" }} opacity={visible ? 1 : 0}>
        <path strokeWidth={strokeWidth} stroke="black" fill="lightgray" d={d.M(edgeStart[0] + normOffsetted + strokeWidth / 2, edgeStart[1] - arrowHeadHeight / 2) + d.l(arrowHeadWidth, arrowHeadHeight / 2) + d.l(-arrowHeadWidth, arrowHeadHeight / 2) + d.z} />
        <rect
            x={edgeStart[0]} y={edgeStart[1] - width / 2}
            width={normOffsetted} height={width} fill="white" stroke="none"
        />
        {
            inEdgeSteps.map(({ start, end, values }, index1) => {
                const s = values.reduce((acc, { value }) => acc + value, 0) * flowScale
                let y = edgeStart[1] - s / 2
                return values.map(({ color, value }, index2) => {
                    const myY = y
                    y += value * flowScale
                    return (
                        <rect
                            key={`${index1}-${index2}`}
                            fill={color} x={edgeStart[0] + normOffsetted - end / transitTime * normOffsetted} y={myY} width={(end - start) / transitTime * normOffsetted} height={value * flowScale} />
                    )
                })
            }).flat()
        }
        <g mask={`url(#${svgIdPrefix}fade-mask)`}>
            {
                queueSteps.map(({ start, end, values }, index1) => {
                    let x = edgeStart[0] - width
                    return values.slice(0).reverse().map(({ color, value }, index2) => {
                        const myX = x
                        x += value * flowScale
                        return (
                            <rect key={`${index1}-${index2}`} fill={color} x={myX} y={edgeStart[1] - width + start * waitingTimeScale}
                                width={value * flowScale} height={(end - start) * waitingTimeScale} />
                        )
                    })
                }).flat()
            }
        </g>
        <path stroke="gray" strokeWidth={strokeWidth} style={{ transition: "opacity 0.2s" }} opacity={queueSteps.length > 0 ? 1 : 0}
            fill="none" d={d.M(edgeStart[0], edgeStart[1]) + d.c(-width / 2, 0, -width / 2, 0, - width / 2, - width)} />
        <rect
            x={edgeStart[0] - strokeWidth / 2} y={edgeStart[1] - width / 2 - strokeWidth / 2}
            width={normOffsetted + strokeWidth} height={width + strokeWidth} stroke="black" strokeWidth={strokeWidth} fill="none"
        />
    </g>
}


export const FlowEdge = ({ flowScale, translate, multiGroup, waitingTimeScale, strokeWidth, svgIdPrefix, from, to, outflowSteps, queue, t, capacity, transitTime, visible = true, offset }) => {
    const { inEdgeSteps, queueSteps } = splitOutflowSteps(outflowSteps, queue, transitTime, capacity, t)

    return (
        <BaseEdge
            strokeWidth={strokeWidth}
            offset={offset}
            translate={translate}
            multiGroup={multiGroup}
            waitingTimeScale={waitingTimeScale}
            svgIdPrefix={svgIdPrefix}
            visible={visible}
            from={from}
            to={to}
            transitTime={transitTime}
            flowScale={flowScale}
            capacity={capacity}
            inEdgeSteps={inEdgeSteps}
            queueSteps={queueSteps} />
    )
}


export const Vertex = ({ label, pos, visible = true, radius = 1, strokeWidth = 0.05 }) => {
    const [cx, cy] = pos
    return <g style={{ transition: "opacity 0.2s" }} opacity={visible ? 1 : 0}>
        <circle cx={cx} cy={cy} r={radius} stroke="black" strokeWidth={strokeWidth} fill="white" />
        {label !== null ? <text x={cx} y={cy} style={{
            textAnchor: 'middle', dominantBaseline: 'central', fontSize: radius,
            userSelect: 'none'
        }}>{label}</text> : null}
    </g>
}

export const ForeignObjectLabel = ({ cx, cy, width = 40, height = 40, children }) => (
    <foreignObject x={cx - width / 2} y={cy - height / 2} width={width} height={height}>
        <div style={{ width, height, display: 'grid', justifyContent: 'center', alignItems: 'center' }}>
            {children}
        </div>
    </foreignObject>
)

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
