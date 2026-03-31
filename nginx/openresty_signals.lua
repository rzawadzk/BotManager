-- ═══════════════════════════════════════════════════════════════════════════════
-- Bot Engine — OpenResty Signal Capture (Lua)
-- ═══════════════════════════════════════════════════════════════════════════════
-- Captures JA4+ TLS fingerprints, HTTP/2 pseudo-header ordering, and HTTP/3
-- (QUIC) transport parameters. Runs in access_by_lua_block before the
-- auth_request subrequest to the scoring engine.
--
-- Installation:
--   1. Place this file in /etc/nginx/lua/openresty_signals.lua
--   2. Add to your nginx.conf http{} block:
--        lua_package_path "/etc/nginx/lua/?.lua;;";
--   3. Use in server{} block via access_by_lua_file or require
--
-- Requires: OpenResty (nginx + LuaJIT) with:
--   - ngx_http_lua_module
--   - lua-resty-core
--   - Optional: lua-resty-openssl for JA4+ computation
--
-- Author: Rafal — VPS Bot Management
-- Version: 2.1 — March 2026
-- ═══════════════════════════════════════════════════════════════════════════════

local _M = {}

-- ─────────────────────────────────────────────────────────────────────────────
-- JA4+ TLS Fingerprint Capture
-- ─────────────────────────────────────────────────────────────────────────────
-- JA4 is a next-gen TLS fingerprint (successor to JA3) that captures:
--   - TLS version, SNI, cipher suites, extensions, ALPN
--   - More resistant to randomization than JA3
--
-- If ngx_http_ssl_ja4_module is compiled in, $ssl_ja4 is available directly.
-- Otherwise, we construct an approximation from available SSL variables.

function _M.capture_ja4(ngx)
    -- Try native JA4 variable first (requires patched OpenSSL/nginx module)
    local ja4 = ngx.var.ssl_ja4
    if ja4 and ja4 ~= "" then
        ngx.req.set_header("X-JA4-Hash", ja4)
        return ja4
    end

    -- Fallback: construct a JA4-like fingerprint from available SSL vars
    local ssl_protocol = ngx.var.ssl_protocol or ""     -- TLSv1.2, TLSv1.3
    local ssl_cipher   = ngx.var.ssl_cipher or ""       -- e.g. TLS_AES_256_GCM_SHA384
    local ssl_curves   = ngx.var.ssl_curves or ""       -- e.g. X25519:P-256
    local alpn         = ngx.var.ssl_alpn_protocol or "" -- h2, http/1.1

    -- Map TLS version to JA4 prefix
    local tls_map = {
        ["TLSv1"]   = "t10",
        ["TLSv1.1"] = "t11",
        ["TLSv1.2"] = "t12",
        ["TLSv1.3"] = "t13",
    }
    local tls_ver = tls_map[ssl_protocol] or "t00"

    -- Build JA4 approximation: tls_version + alpn + cipher + curves
    local components = tls_ver .. "_" .. alpn .. "_" .. ssl_cipher .. "_" .. ssl_curves
    local ja4_approx = ngx.md5(components)

    ngx.req.set_header("X-JA4-Hash", ja4_approx)
    return ja4_approx
end


-- ─────────────────────────────────────────────────────────────────────────────
-- HTTP/2 Pseudo-Header Order Fingerprint
-- ─────────────────────────────────────────────────────────────────────────────
-- HTTP/2 frames include pseudo-headers (:method, :path, :scheme, :authority)
-- in a specific order. Browsers have consistent ordering; automation tools
-- often differ.
--
-- Also captures the order of regular headers for fingerprinting.
-- We use ngx.req.get_headers() which returns headers in order on OpenResty.

function _M.capture_h2_fingerprint(ngx)
    -- Check if this is an HTTP/2 connection
    local http_ver = ngx.req.http_version()
    if http_ver < 2.0 then
        ngx.req.set_header("X-H2-Fingerprint", "")
        return ""
    end

    -- Capture header order
    -- ngx.req.raw_header() gives the raw request headers
    local raw = ngx.req.raw_header(true) -- true = no request line
    if not raw then
        return ""
    end

    local header_names = {}
    for line in raw:gmatch("[^\r\n]+") do
        local name = line:match("^([^:]+):")
        if name then
            table.insert(header_names, name:lower())
        end
    end

    local order_str = table.concat(header_names, "|")
    local fp = ngx.md5(order_str)

    ngx.req.set_header("X-H2-Fingerprint", fp)
    ngx.req.set_header("X-Header-Order", order_str)
    return fp
end


-- ─────────────────────────────────────────────────────────────────────────────
-- HTTP/3 (QUIC) Transport Parameter Capture
-- ─────────────────────────────────────────────────────────────────────────────
-- HTTP/3 over QUIC exposes transport parameters that can fingerprint clients:
--   - initial_max_data, initial_max_stream_data
--   - max_idle_timeout, max_udp_payload_size
--   - active_connection_id_limit
--
-- These are exposed via nginx QUIC variables when using nginx-quic builds.

function _M.capture_h3_params(ngx)
    -- Check if this is HTTP/3
    local http_ver = ngx.req.http_version()
    if http_ver < 3.0 then
        ngx.req.set_header("X-H3-Params", "")
        return ""
    end

    -- Attempt to read QUIC transport parameters
    -- These variables are available in nginx-quic (1.25+) builds
    local quic_vars = {
        max_data     = ngx.var.quic_initial_max_data or "",
        max_stream   = ngx.var.quic_initial_max_stream_data_bidi_local or "",
        idle_timeout = ngx.var.quic_max_idle_timeout or "",
        max_payload  = ngx.var.quic_max_udp_payload_size or "",
        conn_id_lim  = ngx.var.quic_active_connection_id_limit or "",
    }

    -- Build a fingerprint string
    local parts = {}
    for k, v in pairs(quic_vars) do
        if v ~= "" then
            table.insert(parts, k .. "=" .. v)
        end
    end

    local h3_str = table.concat(parts, ";")
    if h3_str == "" then
        -- Fallback: just note it's HTTP/3 without detailed params
        h3_str = "h3=true"
    end

    ngx.req.set_header("X-H3-Params", h3_str)
    return h3_str
end


-- ─────────────────────────────────────────────────────────────────────────────
-- Connection Timing Signals
-- ─────────────────────────────────────────────────────────────────────────────
-- Capture connection-level timing for temporal analysis

function _M.capture_timing(ngx)
    -- Request start time (high resolution)
    local req_start = ngx.now()

    -- Connection reuse indicator
    local conn_requests = ngx.var.connection_requests or "1"

    -- Time since connection was established
    -- (only meaningful for keepalive connections)
    local conn_time = ngx.var.connection_time or "0"

    local timing_str = conn_requests .. ":" .. conn_time .. ":" ..
                       string.format("%.3f", req_start)

    ngx.req.set_header("X-Conn-Timing", timing_str)
    return timing_str
end


-- ─────────────────────────────────────────────────────────────────────────────
-- Main: Capture all signals in one call
-- ─────────────────────────────────────────────────────────────────────────────
-- Usage in nginx.conf:
--   access_by_lua_block {
--       local signals = require "openresty_signals"
--       signals.capture_all(ngx)
--   }

function _M.capture_all(ngx)
    -- Wrap each capture in pcall to prevent any single failure from
    -- blocking the request (fail-open principle)
    local ok, err

    ok, err = pcall(_M.capture_ja4, ngx)
    if not ok then
        ngx.log(ngx.WARN, "bot-engine: JA4 capture failed: ", err)
    end

    ok, err = pcall(_M.capture_h2_fingerprint, ngx)
    if not ok then
        ngx.log(ngx.WARN, "bot-engine: H2 fingerprint failed: ", err)
    end

    ok, err = pcall(_M.capture_h3_params, ngx)
    if not ok then
        ngx.log(ngx.WARN, "bot-engine: H3 params failed: ", err)
    end

    ok, err = pcall(_M.capture_timing, ngx)
    if not ok then
        ngx.log(ngx.WARN, "bot-engine: timing capture failed: ", err)
    end
end


return _M
