type IdType = number | string
type NodeId = IdType
type EdgeId = IdType
type CommodityId = IdType

export class Network {
    nodesMap: { [id : NodeId]: NetNode }
    edgesMap : { [id: EdgeId]: Edge }
    commoditiesMap: { [id: CommodityId]: Commodity}

    constructor(nodesMap: { [id : NodeId]: NetNode }, edgesMap: { [id: EdgeId]: Edge }, commoditiesMap: { [id: CommodityId]: Commodity}) {
        this.nodesMap = nodesMap
        this.edgesMap = edgesMap
        this.commoditiesMap = commoditiesMap
    }

    static fromJson(json: any) {
        const nodes: NetNode[] = json["nodes"]
        const edges: Edge[] = json["edges"]
        const commodities: Commodity[] = json["commodities"]
        const nodesMap = nodes.reduce<{ [id: NodeId]: NetNode }>((acc, node) => {
            acc[node.id] = node
            return acc
        }, {})
        const edgesMap = edges.reduce<{ [id: EdgeId]: Edge }>((acc, edge) => {
            acc[edge.id] = edge
            return acc
        }, {})
        const commoditiesMap = commodities.reduce<{ [id: CommodityId]: Commodity }>((acc, comm) => {
            acc[comm.id] = comm
            return acc
        }, {})
        return new Network(nodesMap, edgesMap, commoditiesMap)
    }
}

export interface NetNode {
    id: NodeId
    label?: string
    x: number
    y: number
}

export interface Edge {
    id: EdgeId
    from: NodeId
    to: NodeId
    capacity: number
    transitTime: number    
}

export interface Commodity {
    id: CommodityId
    color: string    
}