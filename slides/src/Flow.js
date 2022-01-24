class Flow {
    constructor(inflow, outflow, queues) {
        this.inflow = inflow
        this.outflow = outflow
        this.queues = queues
    }
}

class Network {
    constructor(nodes, edges) {
        this.nodesMap = nodes.reduce((acc, node) => {
            acc[node.id] = node
            return acc
        }, {})
        this.edges = edges
    }
}

class GraphNode {
    constructor(id, label, x, y) {
        this.id = id
        this.label = label
        this.x = x
        this.y = y
    }
}

class Edge {
    constructor(id, idFrom, idTo, capacity) {
        this.id = id
        this.idFrom = idFrom
        this.idTo = idTo
        this.capacity = capacity
    }
}

class RightConstant {
    constructor(times, values) {
        this.times = times
        this.values = values
    }

    eval(at) {
        const rnk = elemLRank(this.times, at)
        return this._evalWithLRank(at, rnk)
    }

    _evalWithLRank(at, rnk) {
        if (rnk === -1) {
            return this.values[0]
        } else {
            return this.values[rnk]
        }
    }
}

function elemLRank(arr, x) {
    if (x < arr[0]) { return -1 }
    let low = 0
    let high = arr.length
    while (high > low) {
        const mid = Math.floor((high + low) / 2)
        if (x < arr[mid]) {
            high = mid
        } else { // arr[mid] <= x
            low = mid + 1
        }
    }
    return high - 1
}

class PiecewiseLinear {
    constructor(times, values, lastSlope, firstSlope) {
        this.times = times
        this.values = values
        this.lastSlope = lastSlope
        this.firstSlope = firstSlope
    }

    eval(at) {
        const rnk = elemRank(this.times, at)
        return this._evalWithRank(at, rnk)
    }

    gradient(rnk) {
        if (rnk === -1) return this.firstSlope
        if (rnk === this.times.length - 1) return this.lastSlope
        return (this.values[rnk + 1] - this.values[rnk]) / (this.times[rnk + 1] - this.times[rnk])

    }

    _evalWithRank(at, rnk) {
        if (rnk === -1) {
            const first_grad = this.gradient(rnk)
            return this.values[0] + (at - this.times[0]) * first_grad
        } else if (rnk === this.times.length - 1) {
            const last_grad = this.gradient(rnk)
            return this.values[this.values.length - 1] + (at - this.times[this.times.length - 1]) * last_grad
        }
        return this.values[rnk] + (at - this.times[rnk]) * this.gradient(rnk)

    }

}

function elemRank(arr, x) {
    if (x <= arr[0]) {
        return -1
    }

    let low = 0
    let high = arr.length
    while (high > low) {
        const mid = Math.floor((high + low) / 2)
        if (x <= arr[mid]) {
            high = mid
        } else {
            low = mid + 1
        }
    }
    return high - 1
}
