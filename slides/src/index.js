import React, { useRef } from 'react';
import ReactDOM from 'react-dom';
import LaTex from '@matejmazur/react-katex'
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
  Box,
  Appear
} from 'spectacle';
import { animated, useSpring, useChain } from 'react-spring';
import { Example1Svg } from './example1';


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
    text: '18px',
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
    return <Slide><FlexBox width="1" flexDirection="row" justifyContent="space-between" borderBottom="1px solid black">
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>
        {TITLE}<span style={{ padding: "0 10px" }}> — </span>{section}
      </Text>
      <Text fontFamily="head" fontSize="head" margin="0px" padding="0px" style={{ letterSpacing: "-.5px" }}>{PRESENTER}</Text>
    </FlexBox>
    {children}
    </Slide>
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

const Presentation = () => (
  <Deck theme={theme} template={template}>
    <Slide>
      <Heading>{TITLE}</Heading>
      <Text className="authors" textAlign="center" fontSize="h2">Lukas Graf<sup>1</sup>, Tobias Harks<sup>1</sup>, Kostas Kollias<sup>2</sup>, and Michael Markl<sup>1</sup>
        <div style={{ fontSize: "0.8em", margin: "2em 0", display: "flex", justifyContent: "center" }}><span style={{ width: "300px" }}><b>1</b>: University of Augsburg</span><span style={{ width: "300px" }}><b>2</b>: Google</span></div>
      </Text>
    </Slide>

    <CustomSlide intro section="I. The Flow Model">
      <SubHeading textAlign="left">The Physical Flow Model</SubHeading>
      <Box>
        <div>
          <Box style={{ float: "right" }}>
            <Example1Svg />
          </Box>
          <UnorderedList style={{ margin: "0" }}>
            <ListItem>Directed graph {TeX`G=(V,E)`}</ListItem>
            <ListItem>Edge travel time {TeX`\tau_e > 0`} and edge capacity {TeX`\nu_e> 0`} for {TeX`e\in E`}</ListItem>
            <ListItem>Commodities {TeX`i\in I`} with source and sink {TeX`s_i, t_i\in V`} and <br />network inflow rate {TeX`u_i: \mathbb R_{\geq 0} \to \mathbb R_{\geq 0}`}</ListItem>
          </UnorderedList>
          <Appear><Definition>
            A <i>dynamic flow</i> {TeX`f=(f^+, f^-)`} consists of
            <UnorderedList style={{ margin: "0" }}>
              <ListItem>edge inflow rates {TeX`f^+_{i,e}:\mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {TeX`i\in I, e\in E`} and</ListItem>
              <ListItem>edge outflow rates {TeX`f^-_{i,e}: \mathbb R_{\geq 0}\to \mathbb R_{\geq 0}`} for {TeX`i\in I, e\in E`}.</ListItem>
            </UnorderedList>
          </Definition></Appear>
          <Appear><Notation>
            {TeX`f_e^+ \coloneqq \sum_{i\in I} f_{i,e}^+,`}
            <Appear tagName='span'>{TeX`~~f_e^- \coloneqq \sum_{i\in I} f_{i,e}^-,`}</Appear>
            <Appear tagName='span'>{TeX`~~q_e(\theta) \coloneqq \int_0^\theta f^+_e(z) - f^-_e(z+\tau_e) \,\mathrm dz`}</Appear>
          </Notation></Appear>
          <Appear><Definition>
            A dynamic flow {TeX`f`} is <i>feasible</i> if it fulfills the following conditions:
            <UnorderedList style={{ margin: "0" }}>
              <ShowcaseFormula text="Flow is conserved:" formula={
                BTeX`\sum_{e\in\delta_v^+} f^+_{i,e}(\theta) - \sum_{e\in\delta_v^-} f^-_{i,e}(\theta) 
              \begin{cases}
              = u_i(\theta), & \text{if $v = s_i$}, \\
              = 0, & \text{if $v \notin \{s_i, t_i \}$}, \\
              \leq 0, & \text{if $v = t_i$}.
              \end{cases}`
              } />
              <ShowcaseFormula text="Queues operate at capacity:" formula={BTeX`f_e^-(\theta) = \begin{cases}
            \nu_e,&\text{if $q_e(\theta - \tau_e) > 0$,} \\
            \min\{ f_e^+(\theta- \tau_e), \nu_e \}, &\text{otherwise.}
          \end{cases}`} />
              <ShowcaseFormula text="Capacity is split fairly:" formula={BTeX`
                f_{i,e}^-(\theta) = f_e^-(\theta) \cdot \frac{f_{i,e}^+(\xi)}{f_e^+(\xi)}
                \quad\text{for $\xi\coloneqq \min\{\xi\leq\theta \mid \xi + \tau_e + \frac{q_e(\xi)}{\nu_e} = \theta \}$ with $f_e^+(\xi) > 0$}`} />
            </UnorderedList>
          </Definition></Appear>

        </div>
      </Box>
    </CustomSlide>

    <CustomSlide section="I. The Flow Model">
      <SubHeading textAlign="left">The Behavioral Model</SubHeading>
      <Box>
      </Box>
    </CustomSlide>

  </Deck >
);

const ShowcaseFormula = ({ formula, text }) => {
  return <Stepper values={[true, false]}>
    {(value, step, isActive) => {
      return <ListItem style={{ display: value === false ? 'list-item' : 'block' }}>
        <div style={
          {
            display: 'flex', flexDirection: 'row', alignItems: 'center', height: '30px', transition: 'transform 0.2s',
            transform: value === false ? 'translateY(0px)' : 'translateY(20px)'
          }}>
          <div>{text}</div>
          <div style={{ paddingLeft: '15px', transition: 'transform 0.2s', transform: value === false ? 'scale(.5)' : 'scale(1)', transformOrigin: 'left' }}>{formula}</div>
        </div></ListItem>
    }}
  </Stepper>
}

const TeX = (template) => {
  return <LaTex>{String.raw(template)}</LaTex>
}

const BTeX = (template) => {
  return <LaTex block>{String.raw(template)}</LaTex>
}

const Notation = ({ children }) => {
  return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
    <span><i>Notation. </i></span>
    {children}
  </Box>
}

const Definition = ({ children }) => {
  return <Box margin="32px" style={{ fontSize: theme.fontSizes.text, fontFamily: "Open Sans" }}>
    <span><b>Definition. </b></span>
    {children}
  </Box>

}

ReactDOM.render(<Presentation />, document.getElementById('root'));
