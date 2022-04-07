import { Alignment, Button, ButtonGroup, Icon, Navbar, NavbarGroup, NavbarHeading, Slider } from "@blueprintjs/core"
import * as React from "react"
import { ReactNode, useRef } from "react"
import TeX from '@matejmazur/react-katex'

import styled from 'styled-components'
import useSize from '@react-hook/size'
import { flow, network } from "./sample"
import { Network } from "./Network"
import { Flow } from "./Flow"
import * as _ from "lodash"
import { max, min } from "lodash"
import { calcOutflowSteps, FlowEdge, SvgDefs, Vertex } from "./DynFlowSvg"

const MyContainer = styled.div`
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
`;

const useMinMaxTime = (flow: Flow) => React.useMemo(
    () => {
        const allTimes = _.concat(
            _.map(flow.inflow.flat(), pwc => pwc.times).flat(),
            _.map(flow.outflow.flat(), pwc => pwc.times).flat(),
            _.map(flow.queues, pwc => pwc.times).flat()
        ) 

        return [min(allTimes), max(allTimes)]
    }
, [flow])

const useInitialBoundingBox = (network: Network, flow: Flow) => React.useMemo(
    () => {
        const allXCoordinates = _.map(network.nodesMap, node => node.x)
        const x0 = min(allXCoordinates)
        const x1 = max(allXCoordinates)
        const allYCoordinates = _.map(network.nodesMap, node => node.y)
        const y0 = min(allYCoordinates)
        const y1 = max(allYCoordinates)
        return {
            x0, x1, width: x1 - x0,
            y0, y1, height: y1 - y0
        }
    }, [network, flow]
)

const useAverageEdgeDistance = (network: Network) => React.useMemo(
    () => {
        const distances = _.map(network.edgesMap, (edge, id) => {
            const from = network.nodesMap[edge.from]
            const to = network.nodesMap[edge.to]

            return Math.sqrt((from.x - to.x)**2 + (from.y - to.y)**2)
        })
        return _.sum(distances) / distances.length
    }, [network]
)

const useAverageCapacity = (network: Network) => React.useMemo(
    () => {
        const capacities = _.map(network.edgesMap, (edge, id) => edge.capacity)
        return _.sum(capacities) / capacities.length
    }, [network]
)

const DynamicFlowViewer = (props: { network: Network, flow: Flow }) => {
    const [t, setT] = React.useState(0)
    const [nodeScale, setNodeScale] = React.useState(0.1)
    const avgEdgeDistance = useAverageEdgeDistance(network)
    const avgCapacity = useAverageCapacity(network)

    // avgEdgeWidth / avgEdgeDistance !=! 1/2
    // avgEdgeWidth = ratesScale * avgCapacity
    // => ratesScale = avgEdgeWidth/avgCapacity = avgEdgeDistance / (10 * avgCapacity)
    const initialFlowScale = avgEdgeDistance / (10* avgCapacity)
    const [flowScale, setFlowScale] = React.useState(initialFlowScale)

    const nodeRadius = nodeScale * avgEdgeDistance
    const strokeWidth = 0.05 * nodeRadius
    const edgeOffset = (nodeScale + 0.1) * avgEdgeDistance
    // nodeScale * avgEdgeLength is the radius of a node
    const [minT, maxT] = useMinMaxTime(flow)
    const svgContainerRef = useRef(null);
    const [width, height] = useSize(svgContainerRef);
    const bb = useInitialBoundingBox(network, flow)
    return <>
        <div style={{ display: 'flex', padding: '16px', alignItems: 'center', overflow: 'hidden' }}>
            <Icon icon={'time'} /><div style={{ paddingLeft: '8px', width: '150px', paddingRight: '16px'}}>Time:</div> 
            <Slider onChange={(value) => setT(value)} value={t} min={minT} max={maxT} labelStepSize={(maxT - minT) / 10} stepSize={(maxT-minT)/400} labelPrecision={2} />
        </div>
        <div style={{ display: 'flex', padding: '16px', alignItems: 'center', overflow: 'hidden' }}>
            <Icon icon={'circle'} /><div style={{ paddingLeft: '8px', width: '150px', paddingRight: '16px'}}>Node-Scale:</div> 
            <Slider onChange={(value) => setNodeScale(value)} value={nodeScale} min={0} max={0.5} stepSize={0.01} labelStepSize={1 / 10} />
        </div>
        <div style={{ display: 'flex', padding: '16px', alignItems: 'center', overflow: 'hidden' }}>
            <Icon icon={'flow-linear'} /><div style={{ paddingLeft: '8px', width: '150px', paddingRight: '16px'}}>Flow-Scale:</div> 
            <Slider onChange={(value) => setFlowScale(value)} value={flowScale} min={0} max={10*initialFlowScale} stepSize={initialFlowScale/100} labelPrecision={2}/>
        </div>
        <div style={{flex: 1, position: "relative", overflow: "hidden"}} ref={svgContainerRef}>
            <svg width={width} height={height} viewBox={`${bb.x0 - bb.width} ${bb.y0 - bb.height} ${3*bb.width} ${3*bb.height}`} style={{position: "absolute", top:"0", left: "0", background: "#eee"}}>
                <SvgContent flowScale={flowScale} nodeRadius={nodeRadius} strokeWidth={strokeWidth} edgeOffset={edgeOffset}  t={t} network={network} flow={flow} />
            </svg>
        </div>
    </>
}


export const SvgContent = (
    {t = 0, network, flow, nodeRadius, edgeOffset, strokeWidth, flowScale} :
    {t: number, network: Network, flow: Flow, nodeRadius: number, edgeOffset: number, strokeWidth: number, flowScale: number}
) => {
    const svgIdPrefix = ""
    const outflowSteps = React.useMemo(
        () => flow.outflow.map((outflow: any) => calcOutflowSteps(outflow, network.commoditiesMap)),
        [flow]
    )
    return <>
        <SvgDefs svgIdPrefix={svgIdPrefix} />
        {
            _.map(network.edgesMap, (value, id) => {
                const fromNode = network.nodesMap[value.from]
                const toNode = network.nodesMap[value.to]
                return <FlowEdge
                    strokeWidth={strokeWidth} flowScale={flowScale}
                    key={id} t={t} capacity={value.capacity} offset={edgeOffset} from={[fromNode.x, fromNode.y]} to={[toNode.x, toNode.y]} svgIdPrefix={svgIdPrefix}
                    outflowSteps={outflowSteps[id]} transitTime={value.transitTime} queue={flow.queues[id]}
                />
            })
        }
        {
            _.map(network.nodesMap, (value, id) => {
                return <Vertex key={id} strokeWidth={strokeWidth} radius={nodeRadius} pos={[value.x, value.y]} label={<TeX>{value.label ?? value.id}</TeX>} />
            })
        }
    </>
}



export default () => {
    return <MyContainer>
        <Navbar>
            <NavbarGroup>
                <NavbarHeading>Dynamic Flow Visualization</NavbarHeading>
            </NavbarGroup>
            <NavbarGroup align={Alignment.RIGHT}>
                <ButtonGroup><Button intent="primary" icon="folder-shared" onClick={() => alert("This button has no function yet.")}>Open Dynamic Flow</Button></ButtonGroup>
            </NavbarGroup>
        </Navbar>
        <DynamicFlowViewer network={network} flow={flow} />
    </MyContainer>
}; 