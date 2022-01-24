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
import { animated, useSpring, useChain, useTransition } from 'react-spring';
import { FlowModelSvg } from './DynFlowSvg';


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
        {TITLE + " — " + section}
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


const EdgeFromFlow = () => {

}


const steppedFlow = () => {
  return <Stepper values={[0, 5]}>
    {(value, step, isActive) => {
      const [flip, set] = useState(false)
      const { number } = useTransition({
        from: { number: 0 },
        number: 1,
        delay: 200,
        config: config.molasses,
        onRest: () => set(!flip),
      })
      return <div />
    }}
  </Stepper>
}

const Presentation = () => (
  <Deck theme={theme} template={template}>
    <Slide>
      <Heading>{TITLE}</Heading>
      <Text className="authors" textAlign="center" fontSize="h2">Lukas Graf<sup>1</sup>, Tobias Harks<sup>1</sup>, Kostas Kollias<sup>2</sup>, and Michael Markl<sup>1</sup>
        <div style={{ fontSize: "0.8em", margin: "2em 0", display: "flex", justifyContent: "center" }}><span style={{ width: "300px" }}><b>1</b>: University of Augsburg</span><span style={{width: "300px"}}><b>2</b>: Google</span></div>
      </Text>
    </Slide>

    <CustomSlide intro section="I. The Flow Model">
      <SubHeading textAlign="left">The Physical Flow Model</SubHeading>
      <FlexBox flexDirection="row" alignItems="start">
        <UnorderedList style={{ flex: 1 }}>
          <ListItem>A directed graph <TeX>G=(V,E)</TeX></ListItem>
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
