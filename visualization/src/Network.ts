type IdType = number | string
type NodeId = IdType
type EdgeId = IdType
type CommodityId = IdType

export class Network {
    edgesMap : { [id: EdgeId]: Edge }
    nodesMap: { [id : NodeId]: NetNode }
    commoditiesMap: { [id: CommodityId]: Commodity}

    constructor(nodes: NetNode[], edges: Edge[], commodities: Commodity[]) {
        this.nodesMap = nodes.reduce<{ [id: NodeId]: NetNode }>((acc, node) => {
            acc[node.id] = node
            return acc
        }, {})
        this.edgesMap = edges.reduce<{ [id: EdgeId]: Edge }>((acc, edge) => {
            acc[edge.id] = edge
            return acc
        }, {})
        this.commoditiesMap = commodities.reduce<{ [id: CommodityId]: Commodity }>((acc, comm) => {
            acc[comm.id] = comm
            return acc
        }, {})
    }

    static fromJson(json: any) {
        return new Network(json.nodes, json.edges, json.commodities)
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