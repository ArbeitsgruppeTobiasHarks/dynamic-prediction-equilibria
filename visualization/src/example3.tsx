import TeX from '@matejmazur/react-katex'
import { calcOutflowSteps, FlowEdge, SvgDefs, Vertex } from "./DynFlowSvg";
import { Flow } from "./Flow";
import * as React from 'react'
import * as _ from 'lodash'

import example3FlowData from "./example3FlowData.js"
import { Network } from './Network';


export const network = new Network([
    { id: "s", x: 25, y: 250 }, 
    { id: "v1", x: 25+200, y: 250 }, 
    { id: "t", x: 25+400, y: 250 }, 
    { id: "v3", x: 25, y: 250+200 }, 
    { id: "v4", x: 25+400, y: 250+200 }, 
], [
    { id: 0, from: "s", to: "v1", capacity: 20, transitTime: 200 },
    { id: 1, from: "v1", to: "t", capacity: 10, transitTime: 200 },
    { id: 2, from: "s", to: "v3", capacity: 20, transitTime: 200 },
    { id: 3, from: "v3", to: "v4", capacity: 20, transitTime: 400 },
    { id: 4, from: "v4", to: "t", capacity: 20, transitTime: 200 }
])

export const flow = Flow.fromJson(example3FlowData)

const outflowSteps = flow.outflow.map((outflow: any) => calcOutflowSteps(outflow, ['#a00', '#0a0']))

export const SvgContent = ({t = 0}) => {
    const svgIdPrefix = ""
    return <>
        <SvgDefs svgIdPrefix={svgIdPrefix} />
        {
            _.map(network.edgesMap, (value, id) => {
                const fromNode = network.nodesMap[value.from]
                const toNode = network.nodesMap[value.to]
                return <FlowEdge key={id} t={t} capacity={value.capacity} from={[fromNode.x, fromNode.y]} to={[toNode.x, toNode.y]} svgIdPrefix={svgIdPrefix} outflowSteps={outflowSteps[id]} transitTime={value.transitTime} queue={flow.queues[id]} />;
            })
        }
        {
            _.map(network.nodesMap, (value, id) => {
                return <Vertex key={id} pos={[value.x, value.y]} label={<TeX>{value.label ?? value.id}</TeX>} />
            })
        }
    </>
}
