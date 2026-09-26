declare module "plotly.js/lib/core" {
  import type Plotly from "plotly.js";
  const Core: typeof Plotly;
  export default Core;
}

declare module "plotly.js/lib/scatter" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const Scatter: any;
  export default Scatter;
}
