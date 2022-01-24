import React from 'react';

import TeX from '@matejmazur/react-katex'
import { BaseEdge, calcOutflowSteps, splitOutflowSteps, SvgDefs, Vertex } from "./DynFlowSvg";
import { Flow, PiecewiseLinear, RightConstant } from "./Flow";

const transitTime = [200, 200]
const capacity = [20, 10]

const flow = new Flow(
    [[new RightConstant([-1, 0, 50], [0, 10, 0]), new RightConstant([0], [0])], [new RightConstant([-1, 0, 50], [0, 10, 0]), new RightConstant([0], [0])]],
    [[new RightConstant([-1, 200, 250], [0, 10, 0]), new RightConstant([0], [0])], [new RightConstant([-1, 200, 250], [0, 10, 0]), new RightConstant([0], [0])]],
    [new PiecewiseLinear([0], [0], 0, 0), new PiecewiseLinear([0], [0], 0, 0)]
)

const outflowSteps = flow.outflow.map(outflow => calcOutflowSteps(outflow, ['#a00', '#0a0']))

const FlowEdge = ({ from, to, outflowSteps, queue, t, capacity, transitTime }) => {
    const { inEdgeSteps, queueSteps } = splitOutflowSteps(outflowSteps, queue, transitTime, t)

    return <BaseEdge from={from} to={to} width={capacity} inEdgeSteps={inEdgeSteps} queueSteps={queueSteps} />
}

export const Example1Svg = () => {
    const sPos = [25, 100]
    const vPos = [225, 100]
    const tPos = [425, 100]
    const t = 100
    return <svg width={500} height={500}>
        <SvgDefs />
        <FlowEdge outflowSteps={outflowSteps[0]} from={sPos} to={vPos} capacity={capacity[0]} transitTime={transitTime[0]} queue={flow.queues[0]} t={t} />
        <FlowEdge outflowSteps={outflowSteps[1]} from={vPos} to={tPos} capacity={capacity[1]} transitTime={transitTime[1]} queue={flow.queues[1]} t={t} />

        <Vertex pos={sPos} label={<TeX>s</TeX>} />
        <Vertex pos={tPos} label={<TeX>t</TeX>} />
        <Vertex pos={vPos} label={<TeX>v</TeX>} />
    </svg>
}