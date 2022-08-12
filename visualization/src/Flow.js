import { mapValues } from "lodash"

export class Flow {
    constructor(inflow, outflow, queues) {
        this.inflow = inflow
        this.outflow = outflow
        this.queues = queues
    }

    static fromJson(json) {
        const inflow = json["inflow"].map(inflows => mapValues(inflows, rightConstant => RightConstant.fromJson(rightConstant)))
        const outflow = json["outflow"].map(outflows => mapValues(outflows, rightConstant => RightConstant.fromJson(rightConstant)))
        const queues = json["queues"].map(queue => PiecewiseLinear.fromJson(queue))
        return new Flow(inflow, outflow, queues)
    }
}

export class RightConstant {
    constructor(times, values) {
        this.times = times
        this.values = values
    }

    eval(at) {
        const rnk = elemLRank(this.times, at)
        return this._evalWithLRank(rnk)
    }

    _evalWithLRank(rnk) {
        if (rnk === -1) {
            return this.values[0]
        } else {
            return this.values[rnk]
        }
    }

    static fromJson(json) {
        if (!isArrayOfNumbers(json["times"]) || !isArrayOfNumbers(json["values"])) {
            throw TypeError("Could not parse RightConstant.")
        }
        return new RightConstant(json["times"], json["values"])
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

const isArrayOfNumbers = (value) => {
    if (!Array.isArray(value)) return false
    for (const entry of value) {
        if (typeof entry !== "number") return false
    }
    return true
}

export class PiecewiseLinear {
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

    static fromJson(json) {
        if (!isArrayOfNumbers(json["times"]) || !isArrayOfNumbers(json["values"]) || typeof json["lastSlope"] !== 'number' || typeof json["firstSlope"] !== 'number') {
            throw TypeError("Could not parse PiecewiseLinear.")
        }
        return new PiecewiseLinear(json["times"], json["values"], json["lastSlope"], json["firstSlope"])
    }
}

export function elemRank(arr, x) {
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
