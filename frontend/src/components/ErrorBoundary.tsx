import { Component, type ReactNode } from "react";
import { useLang } from "../i18n/LangContext";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  resetKey?: string;
  resetLabel?: string;
  title?: string;
}

interface State {
  hasError: boolean;
  message: string;
}

function DefaultFallback({ title, resetLabel, message, onReset }: {
  title?: string;
  resetLabel?: string;
  message: string;
  onReset: () => void;
}) {
  const { t } = useLang();
  return (
    <div style={{ padding: 20, color: "#f87171", background: "#1e293b", borderRadius: 8, border: "1px solid rgba(248,113,113,0.2)" }}>
      <h3>{title || t.error.somethingWrong}</h3>
      <p style={{ fontSize: 13, color: "#94a3b8" }}>{message}</p>
      <button type="button" onClick={onReset}>{resetLabel || t.error.retry}</button>
    </div>
  );
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, message: "" };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error.message };
  }

  componentDidUpdate(previousProps: Props) {
    if (this.state.hasError && previousProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, message: "" });
    }
  }

  reset = () => {
    this.setState({ hasError: false, message: "" });
  };

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <DefaultFallback
          title={this.props.title}
          resetLabel={this.props.resetLabel}
          message={this.state.message}
          onReset={this.reset}
        />
      );
    }
    return this.props.children;
  }
}
