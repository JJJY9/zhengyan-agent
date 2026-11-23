import { Component, type ErrorInfo, type ReactNode } from "react";

/** 渲染错误兜底：任何页面组件崩溃时显示错误信息，而不是让整个应用白屏。 */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("页面渲染失败", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="page">
          <div className="error">页面渲染失败：{this.state.error.message}</div>
          <div className="row-btns">
            <button className="button" onClick={() => { this.setState({ error: null }); location.reload(); }}>重新加载</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
