import _, { max, min } from "lodash"
import React, { useEffect, useRef, useState } from "react"
import { Flow, RightConstant } from "./Flow"
import { Network } from "./Network"
import useSize from '@react-hook/size'
import { Button, Card, Collapse, Icon, IconName, Slider } from "@blueprintjs/core"
import { calcOutflowSteps, FlowEdge, SvgDefs, Vertex } from "./DynFlowSvg"



const useMinMaxTime = (flow: Flow) => React.useMemo(
    () => {
        const allTimes = _.concat(
            _.map(flow.inflow, flowObject => Object.values(flowObject).map((pwc: RightConstant) => pwc.times).flat()).flat(),
            _.map(flow.outflow, flowObject => Object.values(flowObject).map((pwc: RightConstant) => pwc.times).flat()).flat(),
            _.map(flow.queues, pwc => pwc.times).flat()
        )

        return [min(allTimes), max(allTimes)]
    }
    , [flow])

const useAvgDistanceTransitTimeRatio = (network: Network) => React.useMemo(
    () => {
        const allRatios = _.map(
            network.edgesMap,
            edge => {
                const from = network.nodesMap[edge.from]
                const to = network.nodesMap[edge.to]
                const edgeDistance = Math.sqrt((from.x - to.x) ** 2 + (from.y - to.y) ** 2)
                if (edge.transitTime == 0) return null
                return edgeDistance / edge.transitTime
            }
        ).filter(ratio => typeof ratio === 'number')
        return _.sum(allRatios) / allRatios.length
    },
    [network]
)

const useInitialBoundingBox = (network: Network) => React.useMemo(
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
    }, [network]
)

const useAverageEdgeDistance = (network: Network) => React.useMemo(
    () => {
        const distances = _.map(network.edgesMap, (edge, id) => {
            const from = network.nodesMap[edge.from]
            const to = network.nodesMap[edge.to]

            return Math.sqrt((from.x - to.x) ** 2 + (from.y - to.y) ** 2)
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

const useScaledNetwork = (network: Network, boundingBox: { x0: number, y0: number, width: number, height: number }) => React.useMemo(
    () => {
        const translateX = boundingBox.x0
        const translateY = boundingBox.y0
        const scale = 1000 / Math.max(boundingBox.width, boundingBox.height)

        const scaledNodesMap = _.mapValues(network.nodesMap, node => ({
            id: node.id,
            label: node.label,
            x: (node.x - translateX) * scale,
            y: (node.y - translateY) * scale
        }))
        return new Network(scaledNodesMap, network.edgesMap, network.commoditiesMap)
    }, [network, boundingBox]
)

const ZOOM_MULTIPLER = 1.125

export const DynamicFlowViewer = (props: { network: Network, flow: Flow }) => {
    const [t, setT] = useState(0)
    const [autoplay, setAutoplay] = useState(false)
    const [minT, maxT] = useMinMaxTime(props.flow)
    const [autoplaySpeed, setAutoplaySpeed] = useState((maxT - minT) / 60)
    React.useEffect(() => {
        if (!autoplay || autoplaySpeed === 0) {
            return
        }
        let last_time_ms = Date.now()
        let done = false
        const callback = () => {
            if (done) return
            setT(t => {
                const new_time_ms = Date.now()
                const delta_t = (new_time_ms - last_time_ms) / 1000 * autoplaySpeed
                last_time_ms = new_time_ms
                return Math.min(maxT, t + delta_t)
            })
            requestAnimationFrame(callback)
        }
        requestAnimationFrame(callback)
        return () => { done = true }
    }, [autoplay, autoplaySpeed, maxT])

    const initialNodeScale = 0.1
    const [nodeScale, setNodeScale] = React.useState(initialNodeScale)
    const scaledNetwork = useScaledNetwork(props.network, useInitialBoundingBox(props.network))
    const avgEdgeDistance = useAverageEdgeDistance(scaledNetwork)
    const avgCapacity = useAverageCapacity(scaledNetwork)
    const initialNodeRadius = initialNodeScale * avgEdgeDistance
    const nodeRadius = nodeScale * avgEdgeDistance

    // avgEdgeWidth / avgEdgeDistance !=! 1/2
    // avgEdgeWidth = ratesScale * avgCapacity
    // => ratesScale = avgEdgeWidth/avgCapacity = avgEdgeDistance / (10 * avgCapacity)
    const initialFlowScale = avgEdgeDistance / (10 * avgCapacity)
    const [flowScale, setFlowScale] = useState(initialFlowScale)
    const avgDistanceTransitTimeRatio = useAvgDistanceTransitTimeRatio(scaledNetwork)
    const [waitingTimeScale, setWaitingTimeScale] = useState(avgDistanceTransitTimeRatio)
    const initialStrokeWidth = 0.05 * initialNodeRadius
    const [strokeWidth, setStrokeWidth] = useState(initialStrokeWidth)
    const initialEdgeOffset = (initialNodeScale + 0.1) * avgEdgeDistance
    const [edgeOffset, setEdgeOffset] = useState(initialEdgeOffset)
    // nodeScale * avgEdgeLength is the radius of a node
    const svgContainerRef = useRef(null);
    const [width, height] = useSize(svgContainerRef);
    const bb = useInitialBoundingBox(scaledNetwork)

    const [manualZoom, setManualZoom] = useState<number | null>(null)

    const initialCenter = [bb.x0 + bb.width / 2, bb.y0 + bb.height / 2]

    const [center, setCenter] = useState(initialCenter)
    const stdZoom = Math.min(width / bb.width, height / bb.height) / 1.5
    const zoom = manualZoom == null ? stdZoom : manualZoom

    const [dragMode, setDragMode] = useState(false)
    const handleMouseDown = () => {
        setDragMode(true)
    }

    const handleWheel: React.WheelEventHandler = (event) => {
        setManualZoom(Math.max(stdZoom / 4, zoom * (event.deltaY < 0 ? ZOOM_MULTIPLER : 1 / ZOOM_MULTIPLER)))
    }

    const [viewOptionsOpen, setViewOptionsOpen] = useState(false)

    useEffect(
        () => {
            if (!dragMode) return
            const handleMouseMove = (event: MouseEvent) => {
                event.stopPropagation()
                event.stopImmediatePropagation()
                event.preventDefault()
                const delta = [event.movementX, event.movementY]
                setCenter(center => [center[0] - delta[0] / zoom, center[1] - delta[1] / zoom])
                return false
            }
            const handleMouseUp = (event: MouseEvent) => {
                setDragMode(false)
            }
            window.addEventListener("mousemove", handleMouseMove)
            window.addEventListener("mouseup", handleMouseUp)
            return () => {
                window.removeEventListener("mousemove", handleMouseMove)
                window.removeEventListener("mouseup", handleMouseUp)
            }
        },
        [dragMode]
    )

    const resetViewOptions = () => {
        setNodeScale(initialNodeScale)
        setFlowScale(initialFlowScale)
        setWaitingTimeScale(avgDistanceTransitTimeRatio)
        setStrokeWidth(initialStrokeWidth)
        setEdgeOffset(initialEdgeOffset)
    }

    // Reset parameters when props change
    useEffect(
        () => {
            setAutoplay(false)
            setAutoplaySpeed((maxT - minT) / 60)
            setT(0)
            resetViewOptions()
            setManualZoom(null)
            setCenter(initialCenter)
        },
        [props.network, props.flow]
    )

    const onResetCamera = () => {
        setCenter(initialCenter)
        setManualZoom(null)
    }

    const viewBox = {
        x: center[0] - width / zoom / 2,
        y: center[1] - height / zoom / 2,
        width: width / zoom,
        height: height / zoom
    }

    const viewBoxString = `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`

    return <>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
            <Card style={{ flex: '1' }}>
                <h5>Time Options</h5>
                <SliderOption icon="time" label="Time:" value={t} setValue={setT} initialValue={minT} min={minT} max={maxT} />
                <div style={{ display: 'flex', padding: '8px 16px', alignItems: 'center', overflow: 'hidden' }}>
                    <Icon icon={'play'} /><div style={{ paddingLeft: '8px', width: '150px', paddingRight: '16px' }}>Autoplay:</div>
                    <Button icon={autoplay ? 'pause' : 'play'} onClick={() => setAutoplay(v => !v)} />
                    <div style={{ display: 'flex', alignItems: 'center', textAlign: 'center', padding: "0px 16px" }}>Speed:<br />(time units per second):</div>
                    <div style={{ padding: '0px', flex: 1 }}>
                        <Slider value={autoplaySpeed} labelPrecision={2} onChange={setAutoplaySpeed} stepSize={.01} min={0} max={(maxT - minT) / 10} labelStepSize={(maxT - minT) / 50} />
                    </div>
                </div>
            </Card>
            <Card style={{ flex: '1' }}>
                <div style={{ display: 'flex', flexDirection: 'row' }}>
                    <h5>View Options</h5>
                    <Button icon={viewOptionsOpen ? 'caret-up' : 'caret-down'} minimal onClick={() => setViewOptionsOpen(value => !value)} />
                    <div style={{ flex: '1', display: 'flex', justifyContent: 'right' }}>
                        <Button icon='reset' minimal onClick={resetViewOptions} />
                    </div>
                </div>
                <Collapse isOpen={viewOptionsOpen}>
                    <SliderOption icon="circle" label="Node-Scale:" value={nodeScale} setValue={setNodeScale} initialValue={initialNodeScale} />
                    <SliderOption icon="flow-linear" label="Edge-Scale:" value={flowScale} setValue={setFlowScale} initialValue={initialFlowScale} max={10 * initialFlowScale} />
                    <SliderOption icon="stopwatch" label="Queue-Scale:" value={waitingTimeScale} setValue={setWaitingTimeScale} initialValue={avgDistanceTransitTimeRatio} />
                    <SliderOption icon="horizontal-inbetween" label="Edge-Offset:" value={edgeOffset} setValue={setEdgeOffset} initialValue={initialEdgeOffset} />
                    <SliderOption icon="full-circle" label="Stroke-Width:" value={strokeWidth} setValue={setStrokeWidth} initialValue={initialStrokeWidth} />
                </Collapse>
            </Card>
        </div>
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }} ref={svgContainerRef}>
            <svg width={width} height={height} viewBox={viewBoxString} onMouseDown={handleMouseDown} onWheel={handleWheel}
                style={{ position: "absolute", top: "0", left: "0", background: "#eee", cursor: "default" }}>
                <SvgContent waitingTimeScale={waitingTimeScale} flowScale={flowScale} nodeRadius={nodeRadius} strokeWidth={strokeWidth}
                    edgeOffset={edgeOffset} t={t} network={scaledNetwork} flow={props.flow} />
            </svg>
            <div style={{ position: "absolute", bottom: "16px", right: 0 }}>
                <div style={{ padding: '8px' }}>
                    <Slider value={Math.log(zoom / stdZoom) / Math.log(ZOOM_MULTIPLER)} min={-10} max={10} onChange={value => setManualZoom(stdZoom * Math.pow(ZOOM_MULTIPLER, value))} vertical labelRenderer={false} showTrackFill={false} />
                </div>
                <Button icon="reset" onClick={onResetCamera} />
            </div>
        </div>
    </>
}

const SliderOption = (
    props: { label: string, initialValue: number, icon: IconName, value: number, setValue: (value: number) => void, max?: number, min?: number }
) => {
    const min = props.min ?? 0.
    const max = props.max ?? 2 * props.initialValue
    return <div style={{ display: 'flex', padding: '8px 0px', alignItems: 'center', overflow: 'hidden' }}>
        <Icon icon={props.icon} style={{ marginLeft: '16px' }} />
        <div style={{ padding: '0 16px', width: '150px' }}>{props.label}</div>
        <Slider onChange={props.setValue} value={props.value} min={min} max={max} stepSize={(max - min) / 100} labelStepSize={(max - min) / 10} labelPrecision={2} />
        <Button style={{ marginLeft: '16px' }} icon='reset' minimal onClick={() => props.setValue(props.initialValue)} />
    </div>
}

export const SvgContent = (
    { t = 0, network, flow, nodeRadius, edgeOffset, strokeWidth, flowScale, waitingTimeScale }:
        { t: number, network: Network, flow: Flow, nodeRadius: number, edgeOffset: number, strokeWidth: number, flowScale: number, waitingTimeScale: number }
) => {
    const svgIdPrefix = ""
    return <>
        <SvgDefs svgIdPrefix={svgIdPrefix} />
        <EdgesCoordinator network={network} waitingTimeScale={waitingTimeScale} strokeWidth={strokeWidth} flowScale={flowScale}
            svgIdPrefix={svgIdPrefix} edgeOffset={edgeOffset} flow={flow} t={t} />
        {
            _.map(network.nodesMap, (value, id) => {
                return <Vertex key={id} strokeWidth={strokeWidth} radius={nodeRadius} pos={[value.x, value.y]} label={value.label ?? value.id} />
            })
        }
    </>
}

const EdgesCoordinator = (
    props: {
        network: Network, waitingTimeScale: number, strokeWidth: number,
        flowScale: number, svgIdPrefix: string, edgeOffset: number, flow: Flow, t: number
    }
) => {
    const outflowSteps = React.useMemo(
        () => props.flow.outflow.map((outflow: any) => calcOutflowSteps(outflow, props.network.commoditiesMap)),
        [props.flow]
    )

    const edgesWithViewOpts = React.useMemo(
        () => {
            const grouped = _.groupBy(props.network.edgesMap, ({ from, to }) => JSON.stringify(from < to ? [from, to] : [to, from]))
            return _.map(grouped, group => {
                const sorted = _.sortBy(group, edge => edge.from)
                const totalCapacity = _.sum(group.map(edge => edge.capacity))
                let translate = -totalCapacity * props.flowScale / 2 - props.strokeWidth * (group.length + 1) / 2
                return sorted.map(edge => {
                    const edgeTranslate = translate + edge.capacity * props.flowScale / 2 + props.strokeWidth / 2
                    translate += edge.capacity * props.flowScale + props.strokeWidth
                    return {
                        translate: edgeTranslate * (edge.from < edge.to ? -1 : 1), edge, multiGroup: group.length > 1
                    }
                })
            }).flat()
        },
        [props.network, props.strokeWidth, props.flowScale]
    )

    return <>
        {_.map(edgesWithViewOpts, ({ edge, translate, multiGroup }) => {
            const fromNode = props.network.nodesMap[edge.from]
            const toNode = props.network.nodesMap[edge.to]
            return <FlowEdge
                id={edge.id}
                waitingTimeScale={props.waitingTimeScale} strokeWidth={props.strokeWidth} flowScale={props.flowScale} translate={translate} multiGroup={multiGroup}
                key={edge.id} t={props.t} capacity={edge.capacity} offset={props.edgeOffset} from={[fromNode.x, fromNode.y]} to={[toNode.x, toNode.y]} svgIdPrefix={props.svgIdPrefix}
                outflowSteps={outflowSteps[edge.id]} transitTime={edge.transitTime} queue={props.flow.queues[edge.id]}
            />
        })}
    </>
}