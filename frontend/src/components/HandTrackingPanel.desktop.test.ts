import { describe, expect, it } from 'vitest';
import nativeConfig from '../../src-tauri/tauri.conf.json?raw';
import infoPlist from '../../src-tauri/Info.plist?raw';
import entitlements from '../../src-tauri/Entitlements.plist?raw';

describe('packaged desktop hand tracking prerequisites', () => {
  it('allows local WASM compilation without permitting JavaScript eval or remote scripts', () => {
    const config = JSON.parse(nativeConfig);
    const directives = config.app.security.csp.split(';').map((part: string) => part.trim().split(/\s+/));
    const scriptSources = directives.find(([name]: string[]) => name === 'script-src')?.slice(1);
    expect(scriptSources).toContain("'wasm-unsafe-eval'");
    expect(scriptSources).toContain("'self'");
    expect(scriptSources).not.toContain("'unsafe-eval'");
    expect(scriptSources).not.toContain('*');
    expect(scriptSources).not.toContain('https:');
  });

  it('declares camera use for macOS permission prompts and hardened runtime', () => {
    expect(infoPlist).toMatch(/<key>NSCameraUsageDescription<\/key>\s*<string>[^<]+<\/string>/);
    expect(entitlements).toMatch(/<key>com\.apple\.security\.device\.camera<\/key>\s*<true\s*\/>/);
  });
});
