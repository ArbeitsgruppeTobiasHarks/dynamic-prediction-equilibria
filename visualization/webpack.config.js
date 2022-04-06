'use strict';

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

// Customized babel loader with the minimum we need to get `mdx` libraries
// working, which unfortunately codegen JSX instead of JS.
const babelLoader = {
  loader: require.resolve('babel-loader'),
  options: {
    // Use user-provided .babelrc
    babelrc: true,
    // ... with some additional needed options.
    presets: [require.resolve('@babel/preset-react')]
  }
};

/**
 * Base configuration for the CLI, core, and examples.
 */

module.exports = (env) => {
  let entry = './src/index.js'
  let bundleOutputFilename = 'deck.js'
  let htmlOutputFilename = 'index.html'
  if (env.poster) {
    entry = './src/poster.js'
    bundleOutputFilename = 'poster.js'
    htmlOutputFilename = 'poster.html'
  }
  return {
    mode: "development",
    entry, // Default for boilerplate generation.
    output: {
      path: path.resolve('dist'),
      filename: bundleOutputFilename
    },
    devtool: 'source-map',
    module: {
      // Not we use `require.resolve` to make sure to use the loader installed
      // within _this_ project's `node_modules` traversal tree.
      rules: [
        {
          test: /\.jsx?$/,
          use: [babelLoader]
        },
        // `.md` files are processed as pure text.
        {
          test: /\.md$/,
          use: [require.resolve('raw-loader')]
        },
        {
          test: /\.(png|svg|jpg|gif)$/,
          use: [require.resolve('file-loader')]
        },
        {
          test: /\.css$/i,
          use: ["style-loader", "css-loader"],
        },
        {
          test: /\.tsx?$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
      ]
    },
    // Default for boilerplate generation.
    plugins: [
      new HtmlWebpackPlugin({
        title: 'Machine-Learned Prediction Equilibrium for Dynamic Traffic Assignment',
        template: './src/index.html',
        filename: htmlOutputFilename
      })
    ],
    resolve: {
      extensions: ['.tsx', '.ts', '.js'],
      fallback: { "util": false, "assert": false }
    }
  }
};
