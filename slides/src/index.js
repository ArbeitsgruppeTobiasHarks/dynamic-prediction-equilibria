import React, { useRef } from 'react';
import ReactDOM from 'react-dom';
import TeX from '@matejmazur/react-katex'
import {
  FlexBox,
  Heading,
  UnorderedList,
  ListItem,
  Progress,
  Stepper,
  Slide,
  Deck,
  Text,
  Box
} from 'spectacle';
import { animated, useSpring, useChain } from 'react-spring';


const theme = {
  fonts: {
    header: '"Open Sans", Helvetica, Arial, sans-serif',
    text: '"Open Sans", Helvetica, Arial, sans-serif',
    head: '"Open Sans", Helvetica, Arial, sans-serif'
  },
  fontWeights: {
    bold: 600
  },
  colors: { // https://material.io/resources/color/#!/?view.left=0&view.right=1&primary.color=B0BEC5&secondary.color=F57F17
    primary: '#000',
    secondary: '#455a64',
    tertiary: '#f5f5f5'
  },
  fontSizes: {
    h1: '48px',
    h2: '32px',
    h3: '28px',
    head: '16px',
    text: '20px'
  }
}

const template = () => (
  <FlexBox
    justifyContent="space-between"
    position="absolute"
    bottom={0}
    width={1}
    zIndex={1}
  >
    <Box padding="0 1em">
    </Box>
    <Box padding="1em">
      <Progress />
    </Box>
  </FlexBox>
);

const SubHeading = (props) => <Text textAlign="center" fontSize="h3" {...props} />

const TITLE = "Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment"
const PRESENTER = "Michael Markl"

const CustomSlide = ({ section, intro = false, children }) => {
  if (!intro) {
    return <FlexBox width="1" flexDirection="row" justifyContent="space-between" borderBottom="1px solid black">
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-1px" }}>
        {TITLE + (!value ? (" — " + section) : "")}
      </Text>
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>{PRESENTER}</Text>
    </FlexBox>
  }

  return <Slide>
    <Stepper values={['hide']} alwaysVisible>
      {(value, step, isActive) => {
        const bigSectionSpringRef = useRef()
        const bigSectionSpring = useSpring({ ref: bigSectionSpringRef, to: isActive ? { opacity: 0 } : { opacity: 1 } })
        const slideSpringRef = useRef()
        const slideSpring = useSpring({ ref: slideSpringRef, to: isActive ? { opacity: 1 } : { opacity: 0 } })

        useChain(isActive ? [bigSectionSpringRef, slideSpringRef] : [slideSpringRef, bigSectionSpringRef])
        return <>
          <FlexBox width="1" flexDirection="row" justifyContent="space-between" borderBottom="1px solid black">
            <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>
              {TITLE}
              <animated.span style={slideSpring}>
                <span style={{ padding: "0 10px" }}> — </span>
                {section}
              </animated.span>
            </Text>
            <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>{PRESENTER}</Text>
          </FlexBox>

          <FlexBox width={1} bottom={0} top={0} justifyContent="center" alignItems="center" zIndex={-1} position="absolute">
            <Heading>
              <animated.div style={bigSectionSpring}>{section}</animated.div>
            </Heading>
          </FlexBox>
          <animated.div style={slideSpring}>
            {children}
          </animated.div>
        </>
      }}
    </Stepper>

  </Slide>
}

const Node = ({ label, pos }) => {
  const radius = 20
  const [cx, cy] = pos
  return <>
    <circle cx={cx} cy={cy} r={radius} stroke="black" fill="white" />
    {label ? (<foreignObject x={cx - radius} y={cy - radius} width={2 * radius} height={2 * radius}>
      <div style={{ width: 2 * radius, height: 2 * radius, display: 'grid', justifyContent: 'center', alignItems: 'center' }}>
        {label}
      </div></foreignObject>) : null}
  </>
}

const Edge = ({ from, to, width = 10, flowSteps = [] }) => {
  const padding = 40
  const d = [to[0] - from[0], to[1] - from[1]]
  const norm = Math.sqrt(d[0] ** 2 + d[1] ** 2)
  // start = from + (to - from)/|to - from| * 30
  const pad = [d[0] / norm * padding, d[1] / norm * padding]
  const start = [from[0] + pad[0], from[1] + pad[1]]
  const deg = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI
  //return <path d={`M${start[0]},${start[1]}L${end[0]},${end[1]}`} />
  const scaledNorm = norm - 2 * padding - width
  const scale = scaledNorm / norm

  return <g transform={`rotate(${deg}, ${start[0]}, ${start[1]})`}>
    <path stroke="black" fill="lightgray" d={`M${start[0] + scaledNorm}, ${start[1] - width} l${width}, ${width} l${-width}, ${width} z`} />
    <rect
      x={start[0]} y={start[1] - width / 2}
      width={scaledNorm} height={width} fill="white" stroke="none"
    />
    {
      flowSteps.map(({ from, to, values }) => {
        const s = values.reduce((acc, { value }) => acc + value, 0)
        let y = start[1] - s / 2
        return values.map(({ color, value }) => {
          const myY = y
          y += value
          return (
            <rect fill={color} x={start[0] + from * scale} y={myY} width={(to - from) * scale} height={value} />
          );
        })
      }).flat()
    }
    <rect
      x={start[0]} y={start[1] - width / 2}
      width={scaledNorm} height={width} stroke="black" fill="none"
    />
  </g>

}

const FlowModelSvg = () => {
  const sPos = [25, 25]
  const vPos = [250, 25]
  const tPos = [475, 25]

  const RED = '#a00'
  const GREEN = '#0a0'
  return <svg width={500} height={200}>

    <Edge from={sPos} to={vPos} width={20} />
    <Edge from={vPos} to={tPos} width={10} flowSteps={[
      { from: 20, to: 50, values: [{ color: RED, value: 5 }, { color: GREEN, value: 5 }] }
    ]} />
    <Node pos={sPos} label={<TeX>s</TeX>} />
    <Node pos={vPos} label={<TeX>v</TeX>} />
    <Node pos={tPos} label={<TeX>t</TeX>} />
  </svg>
}

const Presentation = () => (
  <Deck theme={theme} template={template}>
    <Slide>
      <Heading>{TITLE}</Heading>
      <Text textAlign="center" fontSize="h2">Lukas Graf¹, Tobias Harks¹, Kostas Kollias², and Michael Markl¹
        <div style={{ fontSize: "0.8em", margin: "1em 0" }}><b>1</b>: University of Augsburg, <b>2</b>: Google</div>
      </Text>
    </Slide>

    <CustomSlide intro section="I. The Flow Model">
      <SubHeading textAlign="left">The Physical Flow Model</SubHeading>
      <FlexBox flexDirection="row">
        <UnorderedList style={{ flex: 1 }}>
          <ListItem>An undirected graph <TeX>G=(V,E)</TeX></ListItem>
          <ListItem>Edge travel time <TeX>\tau_e > 0</TeX> for <TeX>e\in E</TeX></ListItem>
          <ListItem>Edge capacity <TeX>\nu_e> 0</TeX> for <TeX>e\in E</TeX></ListItem>
        </UnorderedList>
        <Box style={{ flex: 1 }}>
          <FlowModelSvg />
        </Box>
      </FlexBox>
    </CustomSlide>

  </Deck>
);

ReactDOM.render(<Presentation />, document.getElementById('root'));
