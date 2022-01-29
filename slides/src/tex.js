
import React from 'react'
import LaTex from '@matejmazur/react-katex'


export const Tex = (template) => {
    return <LaTex>{String.raw(template)}</LaTex>
}

export const BTex = (template) => {
    return <LaTex block>{String.raw(template)}</LaTex>
}