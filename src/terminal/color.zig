const colorpkg = @This();

const std = @import("std");
const assert = @import("../quirks.zig").inlineAssert;
const x11_color = @import("x11_color.zig");

/// The default palette.
pub const default: Palette = default: {
    var result: Palette = undefined;

    // Named values
    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        result[i] = Name.default(@enumFromInt(i)) catch unreachable;
    }

    // Cube
    assert(i == 16);
    var r: u8 = 0;
    while (r < 6) : (r += 1) {
        var g: u8 = 0;
        while (g < 6) : (g += 1) {
            var b: u8 = 0;
            while (b < 6) : (b += 1) {
                result[i] = .{
                    .r = if (r == 0) 0 else (r * 40 + 55),
                    .g = if (g == 0) 0 else (g * 40 + 55),
                    .b = if (b == 0) 0 else (b * 40 + 55),
                };

                i += 1;
            }
        }
    }

    // Gray ramp
    assert(i == 232);
    assert(@TypeOf(i) == u8);
    while (i > 0) : (i +%= 1) {
        const value = ((i - 232) * 10) + 8;
        result[i] = .{ .r = value, .g = value, .b = value };
    }

    break :default result;
};

/// Palette is the 256 color palette.
pub const Palette = [256]RGB;

/// Mask that can be used to set which palette indexes were set.
pub const PaletteMask = std.StaticBitSet(@typeInfo(Palette).array.len);

/// Generate the 256-color palette from the user's base16 theme colors,
/// terminal background, and terminal foreground.
///
/// Motivation: The default 256-color palette uses fixed, fully-saturated
/// colors that clash with custom base16 themes, have poor readability in
/// dark shades (the first non-black shade jumps to 37% intensity instead
/// of the expected 20%), and exhibit inconsistent perceived brightness
/// across hues of the same shade (e.g., blue appears darker than green).
/// By generating the extended palette from the user's chosen colors,
/// programs can use the richer 256-color range without requiring their
/// own theme configuration, and light/dark switching works automatically.
///
/// The 216-color cube (indices 16–231) is built via trilinear
/// interpolation in OKLAB space over the 8 base colors. The base16
/// palette maps to the 8 corners of a 6×6×6 RGB cube as follows:
///
///   R=0 edge: bg      → base[1] (red)
///   R=5 edge: base[6] → fg
///   G=0 edge: bg/base[6] (via R) → base[2]/base[4] (green/blue via R)
///   G=5 edge: base[1]/fg (via R) → base[3]/base[5] (yellow/magenta via R)
///
/// For each R slice, four corner colors (c0–c3) are interpolated along
/// the R axis, then for each G row two edge colors (c4–c5) are
/// interpolated along G, and finally each B cell is interpolated along B
/// to produce the final color. OKLAB interpolation ensures perceptually
/// uniform brightness transitions across different hues.
///
/// The 24-step grayscale ramp (indices 232–255) is a simple linear
/// interpolation in OKLAB from the background to the foreground,
/// excluding pure black and white (available in the cube at (0,0,0)
/// and (5,5,5)). The interpolation parameter runs from 1/25 to 24/25.
///
/// Fill `skip` with user-defined color indexes to avoid replacing them.
///
/// Reference: https://gist.github.com/jake-stewart/0a8ea46159a7da2c808e5be2177e1783
pub fn generate256Color(
    base: Palette,
    skip: PaletteMask,
    bg: RGB,
    fg: RGB,
) Palette {
    // Convert the background, foreground, and 8 base theme colors into
    // OKLAB space so that all interpolation is perceptually uniform.
    const bg_lab: OkLab = .fromRgb(bg);
    const fg_lab: OkLab = .fromRgb(fg);
    const base8_lab: [8]OkLab = base8: {
        var base8: [8]OkLab = undefined;
        for (0..8) |i| base8[i] = .fromRgb(base[i]);
        break :base8 base8;
    };

    // Start from the base palette so indices 0–15 are preserved as-is.
    var result = base;

    // Build the 216-color cube (indices 16–231) via trilinear interpolation
    // in OKLAB. The three nested loops correspond to the R, G, and B axes
    // of a 6×6×6 cube. For each R slice, four corner colors (c0–c3) are
    // interpolated along R from the 8 base colors, mapping the cube corners
    // to theme-aware anchors (see doc comment for the mapping). Then for
    // each G row, two edge colors (c4–c5) blend along G, and finally each
    // B cell interpolates along B to produce the final color.
    var idx: usize = 16;
    for (0..6) |ri| {
        // R-axis corners: blend base colors along the red dimension.
        const tr = @as(f32, @floatFromInt(ri)) / 5.0;
        const c0: OkLab = .lerp(tr, bg_lab, base8_lab[1]);
        const c1: OkLab = .lerp(tr, base8_lab[2], base8_lab[3]);
        const c2: OkLab = .lerp(tr, base8_lab[4], base8_lab[5]);
        const c3: OkLab = .lerp(tr, base8_lab[6], fg_lab);
        for (0..6) |gi| {
            // G-axis edges: blend the R-interpolated corners along green.
            const tg = @as(f32, @floatFromInt(gi)) / 5.0;
            const c4: OkLab = .lerp(tg, c0, c1);
            const c5: OkLab = .lerp(tg, c2, c3);
            for (0..6) |bi| {
                // B-axis: final interpolation along blue, then convert back to RGB.
                if (!skip.isSet(idx)) {
                    const c6: OkLab = .lerp(
                        @as(f32, @floatFromInt(bi)) / 5.0,
                        c4,
                        c5,
                    );
                    result[idx] = c6.toRgb();
                }

                idx += 1;
            }
        }
    }

    // Build the 24-step grayscale ramp (indices 232–255) by linearly
    // interpolating in OKLAB from background to foreground. The parameter
    // runs from 1/25 to 24/25, excluding the endpoints which are already
    // available in the cube at (0,0,0) and (5,5,5).
    for (0..24) |i| {
        const t = @as(f32, @floatFromInt(i + 1)) / 25.0;
        if (!skip.isSet(idx)) {
            const c: OkLab = .lerp(t, bg_lab, fg_lab);
            result[idx] = c.toRgb();
        }
        idx += 1;
    }

    return result;
}

/// A palette that can have its colors changed and reset. Purposely built
/// for terminal color operations.
pub const DynamicPalette = struct {
    /// The current palette including any user modifications.
    current: Palette,

    /// The original/default palette values.
    original: Palette,

    /// A bitset where each bit represents whether the corresponding
    /// palette index has been modified from its default value.
    mask: PaletteMask,

    pub const default: DynamicPalette = .init(colorpkg.default);

    /// Initialize a dynamic palette with a default palette.
    pub fn init(def: Palette) DynamicPalette {
        return .{
            .current = def,
            .original = def,
            .mask = .initEmpty(),
        };
    }

    /// Set a custom color at the given palette index.
    pub fn set(self: *DynamicPalette, idx: u8, color: RGB) void {
        self.current[idx] = color;
        self.mask.set(idx);
    }

    /// Reset the color at the given palette index to its original value.
    pub fn reset(self: *DynamicPalette, idx: u8) void {
        self.current[idx] = self.original[idx];
        self.mask.unset(idx);
    }

    /// Reset all colors to their original values.
    pub fn resetAll(self: *DynamicPalette) void {
        self.* = .init(self.original);
    }

    /// Change the default palette, but preserve the changed values.
    pub fn changeDefault(self: *DynamicPalette, def: Palette) void {
        self.original = def;

        // Fast path, the palette is usually not changed.
        if (self.mask.count() == 0) {
            self.current = self.original;
            return;
        }

        // There are usually less set than unset, so iterate over the changed
        // values and override them.
        var current = def;
        var it = self.mask.iterator(.{});
        while (it.next()) |idx| current[idx] = self.current[idx];
        self.current = current;
    }
};

/// RGB value that can be changed and reset. This can also be totally unset
/// in every way, in which case the caller can determine their own ultimate
/// default.
pub const DynamicRGB = struct {
    override: ?RGB,
    default: ?RGB,

    pub const unset: DynamicRGB = .{ .override = null, .default = null };

    pub fn init(def: RGB) DynamicRGB {
        return .{
            .override = null,
            .default = def,
        };
    }

    pub fn get(self: *const DynamicRGB) ?RGB {
        return self.override orelse self.default;
    }

    pub fn set(self: *DynamicRGB, color: RGB) void {
        self.override = color;
    }

    pub fn reset(self: *DynamicRGB) void {
        self.override = self.default;
    }
};

/// Color names in the standard 8 or 16 color palette.
pub const Name = enum(u8) {
    black = 0,
    red = 1,
    green = 2,
    yellow = 3,
    blue = 4,
    magenta = 5,
    cyan = 6,
    white = 7,

    bright_black = 8,
    bright_red = 9,
    bright_green = 10,
    bright_yellow = 11,
    bright_blue = 12,
    bright_magenta = 13,
    bright_cyan = 14,
    bright_white = 15,

    // Remainders are valid unnamed values in the 256 color palette.
    _,

    pub const C = u8;

    pub fn cval(self: Name) C {
        return @intFromEnum(self);
    }

    /// Default colors for tagged values.
    pub fn default(self: Name) error{NoDefaultValue}!RGB {
        return switch (self) {
            .black => RGB{ .r = 0x1D, .g = 0x1F, .b = 0x21 },
            .red => RGB{ .r = 0xCC, .g = 0x66, .b = 0x66 },
            .green => RGB{ .r = 0xB5, .g = 0xBD, .b = 0x68 },
            .yellow => RGB{ .r = 0xF0, .g = 0xC6, .b = 0x74 },
            .blue => RGB{ .r = 0x81, .g = 0xA2, .b = 0xBE },
            .magenta => RGB{ .r = 0xB2, .g = 0x94, .b = 0xBB },
            .cyan => RGB{ .r = 0x8A, .g = 0xBE, .b = 0xB7 },
            .white => RGB{ .r = 0xC5, .g = 0xC8, .b = 0xC6 },

            .bright_black => RGB{ .r = 0x66, .g = 0x66, .b = 0x66 },
            .bright_red => RGB{ .r = 0xD5, .g = 0x4E, .b = 0x53 },
            .bright_green => RGB{ .r = 0xB9, .g = 0xCA, .b = 0x4A },
            .bright_yellow => RGB{ .r = 0xE7, .g = 0xC5, .b = 0x47 },
            .bright_blue => RGB{ .r = 0x7A, .g = 0xA6, .b = 0xDA },
            .bright_magenta => RGB{ .r = 0xC3, .g = 0x97, .b = 0xD8 },
            .bright_cyan => RGB{ .r = 0x70, .g = 0xC0, .b = 0xB1 },
            .bright_white => RGB{ .r = 0xEA, .g = 0xEA, .b = 0xEA },

            else => error.NoDefaultValue,
        };
    }
};

/// The "special colors" as denoted by xterm. These can be set via
/// OSC 5 or via OSC 4 by adding the palette length to it.
///
/// https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
pub const Special = enum(u3) {
    bold = 0,
    underline = 1,
    blink = 2,
    reverse = 3,
    italic = 4,

    pub fn osc4(self: Special) u16 {
        // "The special colors can also be set by adding the maximum
        // number of colors (e.g., 88 or 256) to these codes in an
        // OSC 4  control" - xterm ctlseqs
        const max = @typeInfo(Palette).array.len;
        return @as(u16, @intCast(@intFromEnum(self))) + max;
    }

    test "osc4" {
        const testing = std.testing;
        try testing.expectEqual(256, Special.bold.osc4());
        try testing.expectEqual(257, Special.underline.osc4());
        try testing.expectEqual(258, Special.blink.osc4());
        try testing.expectEqual(259, Special.reverse.osc4());
        try testing.expectEqual(260, Special.italic.osc4());
    }
};

test Special {
    _ = Special;
}

/// The "dynamic colors" as denoted by xterm. These can be set via
/// OSC 10 through 19.
pub const Dynamic = enum(u5) {
    foreground = 10,
    background = 11,
    cursor = 12,
    pointer_foreground = 13,
    pointer_background = 14,
    tektronix_foreground = 15,
    tektronix_background = 16,
    highlight_background = 17,
    tektronix_cursor = 18,
    highlight_foreground = 19,

    /// The next dynamic color sequentially. This is required because
    /// specifying colors sequentially without their index will automatically
    /// use the next dynamic color.
    ///
    /// "Each successive parameter changes the next color in the list.  The
    /// value of Ps tells the starting point in the list."
    pub fn next(self: Dynamic) ?Dynamic {
        return std.meta.intToEnum(
            Dynamic,
            @intFromEnum(self) + 1,
        ) catch null;
    }

    test "next" {
        const testing = std.testing;
        try testing.expectEqual(.background, Dynamic.foreground.next());
        try testing.expectEqual(.cursor, Dynamic.background.next());
        try testing.expectEqual(.pointer_foreground, Dynamic.cursor.next());
        try testing.expectEqual(.pointer_background, Dynamic.pointer_foreground.next());
        try testing.expectEqual(.tektronix_foreground, Dynamic.pointer_background.next());
        try testing.expectEqual(.tektronix_background, Dynamic.tektronix_foreground.next());
        try testing.expectEqual(.highlight_background, Dynamic.tektronix_background.next());
        try testing.expectEqual(.tektronix_cursor, Dynamic.highlight_background.next());
        try testing.expectEqual(.highlight_foreground, Dynamic.tektronix_cursor.next());
        try testing.expectEqual(null, Dynamic.highlight_foreground.next());
    }
};

test Dynamic {
    _ = Dynamic;
}

/// RGB
pub const RGB = packed struct(u24) {
    r: u8 = 0,
    g: u8 = 0,
    b: u8 = 0,

    pub const C = extern struct {
        r: u8,
        g: u8,
        b: u8,
    };

    pub fn cval(self: RGB) C {
        return .{
            .r = self.r,
            .g = self.g,
            .b = self.b,
        };
    }

    pub fn eql(self: RGB, other: RGB) bool {
        return self.r == other.r and self.g == other.g and self.b == other.b;
    }

    /// Calculates the contrast ratio between two colors. The contrast
    /// ration is a value between 1 and 21 where 1 is the lowest contrast
    /// and 21 is the highest contrast.
    ///
    /// https://www.w3.org/TR/WCAG20/#contrast-ratiodef
    pub fn contrast(self: RGB, other: RGB) f64 {
        // pair[0] = lighter, pair[1] = darker
        const pair: [2]f64 = pair: {
            const self_lum = self.luminance();
            const other_lum = other.luminance();
            if (self_lum > other_lum) break :pair .{ self_lum, other_lum };
            break :pair .{ other_lum, self_lum };
        };

        return (pair[0] + 0.05) / (pair[1] + 0.05);
    }

    /// Calculates luminance based on the W3C formula. This returns a
    /// normalized value between 0 and 1 where 0 is black and 1 is white.
    ///
    /// https://www.w3.org/TR/WCAG20/#relativeluminancedef
    pub fn luminance(self: RGB) f64 {
        const r_lum = componentLuminance(self.r);
        const g_lum = componentLuminance(self.g);
        const b_lum = componentLuminance(self.b);
        return 0.2126 * r_lum + 0.7152 * g_lum + 0.0722 * b_lum;
    }

    /// Calculates single-component luminance based on the W3C formula.
    ///
    /// Expects sRGB color space which at the time of writing we don't
    /// generally use but it's a good enough approximation until we fix that.
    /// https://www.w3.org/TR/WCAG20/#relativeluminancedef
    fn componentLuminance(c: u8) f64 {
        const c_f64: f64 = @floatFromInt(c);
        const normalized: f64 = c_f64 / 255;
        if (normalized <= 0.03928) return normalized / 12.92;
        return std.math.pow(f64, (normalized + 0.055) / 1.055, 2.4);
    }

    /// Calculates "perceived luminance" which is better for determining
    /// light vs dark.
    ///
    /// Source: https://www.w3.org/TR/AERT/#color-contrast
    pub fn perceivedLuminance(self: RGB) f64 {
        const r_f64: f64 = @floatFromInt(self.r);
        const g_f64: f64 = @floatFromInt(self.g);
        const b_f64: f64 = @floatFromInt(self.b);
        return 0.299 * (r_f64 / 255) + 0.587 * (g_f64 / 255) + 0.114 * (b_f64 / 255);
    }

    comptime {
        assert(@bitSizeOf(RGB) == 24);
        assert(@sizeOf(RGB) == 4);
    }

    /// Parse a color from a floating point intensity value.
    ///
    /// The value should be between 0.0 and 1.0, inclusive.
    fn fromIntensity(value: []const u8) error{InvalidFormat}!u8 {
        const i = std.fmt.parseFloat(f64, value) catch {
            @branchHint(.cold);
            return error.InvalidFormat;
        };
        if (i < 0.0 or i > 1.0) {
            @branchHint(.cold);
            return error.InvalidFormat;
        }

        return @intFromFloat(i * std.math.maxInt(u8));
    }

    /// Parse a color from a string of hexadecimal digits
    ///
    /// The string can contain 1, 2, 3, or 4 characters and represents the color
    /// value scaled in 4, 8, 12, or 16 bits, respectively.
    fn fromHex(value: []const u8) error{InvalidFormat}!u8 {
        if (value.len == 0 or value.len > 4) {
            @branchHint(.cold);
            return error.InvalidFormat;
        }

        const color = std.fmt.parseUnsigned(u16, value, 16) catch {
            @branchHint(.cold);
            return error.InvalidFormat;
        };

        const divisor: usize = switch (value.len) {
            1 => std.math.maxInt(u4),
            2 => std.math.maxInt(u8),
            3 => std.math.maxInt(u12),
            4 => std.math.maxInt(u16),
            else => unreachable,
        };

        return @intCast(@as(usize, color) * std.math.maxInt(u8) / divisor);
    }

    /// Parse a color specification.
    ///
    /// Any of the following forms are accepted:
    ///
    /// 1. rgb:<red>/<green>/<blue>
    ///
    ///    <red>, <green>, <blue> := h | hh | hhh | hhhh
    ///
    ///    where `h` is a single hexadecimal digit.
    ///
    /// 2. rgbi:<red>/<green>/<blue>
    ///
    ///    where <red>, <green>, and <blue> are floating point values between
    ///    0.0 and 1.0 (inclusive).
    ///
    /// 3. #rgb, #rrggbb, #rrrgggbbb #rrrrggggbbbb
    ///
    ///    where `r`, `g`, and `b` are a single hexadecimal digit.
    ///    These specify a color with 4, 8, 12, and 16 bits of precision
    ///    per color channel.
    pub fn parse(value: []const u8) error{InvalidFormat}!RGB {
        if (value.len == 0) {
            @branchHint(.cold);
            return error.InvalidFormat;
        }

        if (value[0] == '#') {
            switch (value.len) {
                4 => return RGB{
                    .r = try RGB.fromHex(value[1..2]),
                    .g = try RGB.fromHex(value[2..3]),
                    .b = try RGB.fromHex(value[3..4]),
                },
                7 => return RGB{
                    .r = try RGB.fromHex(value[1..3]),
                    .g = try RGB.fromHex(value[3..5]),
                    .b = try RGB.fromHex(value[5..7]),
                },
                10 => return RGB{
                    .r = try RGB.fromHex(value[1..4]),
                    .g = try RGB.fromHex(value[4..7]),
                    .b = try RGB.fromHex(value[7..10]),
                },
                13 => return RGB{
                    .r = try RGB.fromHex(value[1..5]),
                    .g = try RGB.fromHex(value[5..9]),
                    .b = try RGB.fromHex(value[9..13]),
                },

                else => {
                    @branchHint(.cold);
                    return error.InvalidFormat;
                },
            }
        }

        // Check for X11 named colors. We allow whitespace around the edges
        // of the color because Kitty allows whitespace. This is not part of
        // any spec I could find.
        if (x11_color.map.get(std.mem.trim(u8, value, " "))) |rgb| return rgb;

        if (value.len < "rgb:a/a/a".len or !std.mem.eql(u8, value[0..3], "rgb")) {
            @branchHint(.cold);
            return error.InvalidFormat;
        }

        var i: usize = 3;

        const use_intensity = if (value[i] == 'i') blk: {
            i += 1;
            break :blk true;
        } else false;

        if (value[i] != ':') {
            @branchHint(.cold);
            return error.InvalidFormat;
        }

        i += 1;

        const r = r: {
            const slice = if (std.mem.indexOfScalarPos(u8, value, i, '/')) |end|
                value[i..end]
            else {
                @branchHint(.cold);
                return error.InvalidFormat;
            };

            i += slice.len + 1;

            break :r if (use_intensity)
                try RGB.fromIntensity(slice)
            else
                try RGB.fromHex(slice);
        };

        const g = g: {
            const slice = if (std.mem.indexOfScalarPos(u8, value, i, '/')) |end|
                value[i..end]
            else {
                @branchHint(.cold);
                return error.InvalidFormat;
            };

            i += slice.len + 1;

            break :g if (use_intensity)
                try RGB.fromIntensity(slice)
            else
                try RGB.fromHex(slice);
        };

        const b = if (use_intensity)
            try RGB.fromIntensity(value[i..])
        else
            try RGB.fromHex(value[i..]);

        return RGB{
            .r = r,
            .g = g,
            .b = b,
        };
    }
};

/// Oklab color space - perceptually uniform for color operations
/// <https://bottosson.github.io/posts/oklab>
/// <https://bottosson.github.io/posts/gamutclipping>
/// <https://bottosson.github.io/posts/colorpicker>
/// <https://raphlinus.github.io/color/2021/01/18/oklab-critique.html>
const OkLab = struct {
    l: f32, // Lightness [0.0, 1.0]
    a: f32, // Green-Red axis [-1.0, 1.0]
    b: f32, // Blue-Yellow axis [-1.0, 1.0]

    // generally k1=0.206, but k=0.25 gives more consistent dark shades
    // compared to CIELAB, which is known for having correct lightness.
    const k1: f32 = 0.25;
    const k2: f32 = 0.03;
    const k3: f32 = (1.0 + k1) / (1.0 + k2);

    fn toe(x: f32) f32 {
        return 0.5 * (k3 * x - k1 + @sqrt((k3 * x - k1) * (k3 * x - k1) + 4.0 * k2 * k3 * x));
    }

    fn toeInv(x: f32) f32 {
        return (x * x + k1 * x) / (k3 * (x + k2));
    }

    fn linearSrgbToOklab(r: f32, g: f32, b: f32) OkLab {
        const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        const l_ = std.math.cbrt(l);
        const m_ = std.math.cbrt(m);
        const s_ = std.math.cbrt(s);

        return .{
            .l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            .a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            .b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        };
    }

    fn oklabToLinearSrgb(L: f32, aa: f32, bb: f32) struct { r: f32, g: f32, b: f32 } {
        const l_ = L + 0.3963377774 * aa + 0.2158037573 * bb;
        const m_ = L - 0.1055613458 * aa - 0.0638541728 * bb;
        const s_ = L - 0.0894841775 * aa - 1.2914855480 * bb;

        const l = l_ * l_ * l_;
        const m = m_ * m_ * m_;
        const s = s_ * s_ * s_;

        return .{
            .r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            .g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            .b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        };
    }

    fn computeMaxSaturation(a_: f32, b_: f32) f32 {
        const k0, const kk1, const kk2, const kk3, const kk4, const wl, const wm, const ws =
            if (-1.88170328 * a_ - 0.80936493 * b_ > 1.0)
                .{
                    @as(f32, 1.19086277),
                    @as(f32, 1.76576728),
                    @as(f32, 0.59662641),
                    @as(f32, 0.75515197),
                    @as(f32, 0.56771245),
                    @as(f32, 4.0767416621),
                    @as(f32, -3.3077115913),
                    @as(f32, 0.2309699292)
                }
            else if (1.81444104 * a_ - 1.19445276 * b_ > 1.0)
                .{
                    @as(f32, 0.73956515),
                    @as(f32, -0.45954404),
                    @as(f32, 0.08285427),
                    @as(f32, 0.12541070),
                    @as(f32, 0.14503204),
                    @as(f32, -1.2684380046),
                    @as(f32, 2.6097574011),
                    @as(f32, -0.3413193965)
                }
            else
                .{
                    @as(f32, 1.35733652),
                    @as(f32, -0.00915799),
                    @as(f32, -1.15130210),
                    @as(f32, -0.50559606),
                    @as(f32, 0.00692167),
                    @as(f32, -0.0041960863),
                    @as(f32, -0.7034186147),
                    @as(f32, 1.7076147010)
                };

        var S = k0 + kk1 * a_ + kk2 * b_ + kk3 * a_ * a_ + kk4 * a_ * b_;

        const k_l = 0.3963377774 * a_ + 0.2158037573 * b_;
        const k_m = -0.1055613458 * a_ - 0.0638541728 * b_;
        const k_s = -0.0894841775 * a_ - 1.2914855480 * b_;

        const l_ = 1.0 + S * k_l;
        const m_ = 1.0 + S * k_m;
        const s_ = 1.0 + S * k_s;

        const l = l_ * l_ * l_;
        const m = m_ * m_ * m_;
        const s = s_ * s_ * s_;

        const l_dS = 3.0 * k_l * l_ * l_;
        const m_dS = 3.0 * k_m * m_ * m_;
        const s_dS = 3.0 * k_s * s_ * s_;

        const l_dS2 = 6.0 * k_l * k_l * l_;
        const m_dS2 = 6.0 * k_m * k_m * m_;
        const s_dS2 = 6.0 * k_s * k_s * s_;

        const f = wl * l + wm * m + ws * s;
        const f1 = wl * l_dS + wm * m_dS + ws * s_dS;
        const f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2;

        S = S - f * f1 / (f1 * f1 - 0.5 * f * f2);
        return S;
    }

    fn findCusp(a_: f32, b_: f32) struct { L: f32, C: f32 } {
        const S_cusp = computeMaxSaturation(a_, b_);
        const rgb = oklabToLinearSrgb(1.0, S_cusp * a_, S_cusp * b_);
        const L_cusp = std.math.cbrt(1.0 / @max(@max(rgb.r, rgb.g), @max(rgb.b, 0.0)));
        return .{ .L = L_cusp, .C = L_cusp * S_cusp };
    }

    fn findGamutIntersection(a_: f32, b_: f32, L1: f32, C1: f32, L0: f32, cusp_L: f32, cusp_C: f32) f32 {
        var t: f32 = undefined;
        if ((L1 - L0) * cusp_C - (cusp_L - L0) * C1 <= 0.0) {
            t = cusp_C * L0 / (C1 * cusp_L + cusp_C * (L0 - L1));
        } else {
            t = cusp_C * (L0 - 1.0) / (C1 * (cusp_L - 1.0) + cusp_C * (L0 - L1));

            const dL = L1 - L0;
            const dC = C1;

            const kl = 0.3963377774 * a_ + 0.2158037573 * b_;
            const km = -0.1055613458 * a_ - 0.0638541728 * b_;
            const ks = -0.0894841775 * a_ - 1.2914855480 * b_;

            const l_dt = dL + dC * kl;
            const m_dt = dL + dC * km;
            const s_dt = dL + dC * ks;

            {
                const L = L0 * (1.0 - t) + t * L1;
                const C = t * C1;

                const l_ = L + C * kl;
                const m_ = L + C * km;
                const s_ = L + C * ks;

                const l = l_ * l_ * l_;
                const m = m_ * m_ * m_;
                const s = s_ * s_ * s_;

                const ldt = 3.0 * l_dt * l_ * l_;
                const mdt = 3.0 * m_dt * m_ * m_;
                const sdt = 3.0 * s_dt * s_ * s_;

                const ldt2 = 6.0 * l_dt * l_dt * l_;
                const mdt2 = 6.0 * m_dt * m_dt * m_;
                const sdt2 = 6.0 * s_dt * s_dt * s_;

                const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1.0;
                const r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt;
                const r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2;
                const u_r = r1 / (r1 * r1 - 0.5 * r * r2);
                const t_r = if (u_r >= 0.0) -r * u_r else 1.0e5;

                const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1.0;
                const g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt;
                const g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2;
                const u_g = g1 / (g1 * g1 - 0.5 * g * g2);
                const t_g = if (u_g >= 0.0) -g * u_g else 1.0e5;

                const b2 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1.0;
                const b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt;
                const b22 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2;
                const u_b = b1 / (b1 * b1 - 0.5 * b2 * b22);
                const t_b = if (u_b >= 0.0) -b2 * u_b else 1.0e5;

                t += @min(t_r, @min(t_g, t_b));
            }
        }
        return t;
    }

    fn linearToSrgb(c: f32) f32 {
        if (c <= 0.0031308) return 12.92 * c;
        return 1.055 * std.math.pow(f32, c, 1.0 / 2.4) - 0.055;
    }

    fn srgbToLinear(c: f32) f32 {
        if (c <= 0.04045) return c / 12.92;
        return std.math.pow(f32, (c + 0.055) / 1.055, 2.4);
    }

    fn fromRgbRaw(rgb: RGB) OkLab {
        const r = @as(f32, @floatFromInt(rgb.r)) / 255.0;
        const g = @as(f32, @floatFromInt(rgb.g)) / 255.0;
        const b = @as(f32, @floatFromInt(rgb.b)) / 255.0;
        return linearSrgbToOklab(srgbToLinear(r), srgbToLinear(g), srgbToLinear(b));
    }

    fn toRgbRaw(self: OkLab) RGB {
        const rgb = oklabToLinearSrgb(self.l, self.a, self.b);
        return .{
            .r = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.r)))) * 255.0 + 0.5),
            .g = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.g)))) * 255.0 + 0.5),
            .b = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.b)))) * 255.0 + 0.5),
        };
    }

    /// Oklab with toe-corrected lightness and preserve-chroma gamut clipping.
    fn fromRgbCorrected(rgb: RGB) OkLab {
        const lab = fromRgbRaw(rgb);
        return .{ .l = toe(lab.l), .a = lab.a, .b = lab.b };
    }

    /// Oklab with toe-corrected lightness and preserve-chroma gamut clipping.
    fn toRgbCorrected(self: OkLab) RGB {
        const L = toeInv(@max(0.0, @min(1.0, self.l)));
        const C = @sqrt(self.a * self.a + self.b * self.b);

        if (C < 1.0e-10) {
            const rgb = oklabToLinearSrgb(L, 0.0, 0.0);
            const grey: u8 = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.r)))) * 255.0 + 0.5);
            return .{ .r = grey, .g = grey, .b = grey };
        }

        const a_ = self.a / C;
        const b_ = self.b / C;

        const cusp = findCusp(a_, b_);
        const t = @min(findGamutIntersection(a_, b_, L, C, L, cusp.L, cusp.C), 1.0);
        const C_clipped = t * C;

        const rgb = oklabToLinearSrgb(L, C_clipped * a_, C_clipped * b_);
        return .{
            .r = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.r)))) * 255.0 + 0.5),
            .g = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.g)))) * 255.0 + 0.5),
            .b = @intFromFloat(@max(0.0, @min(1.0, linearToSrgb(@max(0.0, rgb.b)))) * 255.0 + 0.5),
        };
    }

    pub fn fromRgb(rgb: RGB) OkLab {
        return fromRgbCorrected(rgb);
    }

    pub fn toRgb(self: OkLab) RGB {
        return self.toRgbCorrected();
    }

    pub fn lerp(t: f32, a: OkLab, b: OkLab) OkLab {
        return .{
            .l = a.l + t * (b.l - a.l),
            .a = a.a + t * (b.a - a.a),
            .b = a.b + t * (b.b - a.b),
        };
    }
};

test "palette: default" {
    const testing = std.testing;

    // Safety check
    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        try testing.expectEqual(Name.default(@as(Name, @enumFromInt(i))), default[i]);
    }
}

test "RGB.parse" {
    const testing = std.testing;

    try testing.expectEqual(RGB{ .r = 255, .g = 0, .b = 0 }, try RGB.parse("rgbi:1.0/0/0"));
    try testing.expectEqual(RGB{ .r = 127, .g = 160, .b = 0 }, try RGB.parse("rgb:7f/a0a0/0"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("rgb:f/ff/fff"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("#ffffff"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("#fff"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("#fffffffff"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("#ffffffffffff"));
    try testing.expectEqual(RGB{ .r = 255, .g = 0, .b = 16 }, try RGB.parse("#ff0010"));

    try testing.expectEqual(RGB{ .r = 0, .g = 0, .b = 0 }, try RGB.parse("black"));
    try testing.expectEqual(RGB{ .r = 255, .g = 0, .b = 0 }, try RGB.parse("red"));
    try testing.expectEqual(RGB{ .r = 0, .g = 255, .b = 0 }, try RGB.parse("green"));
    try testing.expectEqual(RGB{ .r = 0, .g = 0, .b = 255 }, try RGB.parse("blue"));
    try testing.expectEqual(RGB{ .r = 255, .g = 255, .b = 255 }, try RGB.parse("white"));

    try testing.expectEqual(RGB{ .r = 124, .g = 252, .b = 0 }, try RGB.parse("LawnGreen"));
    try testing.expectEqual(RGB{ .r = 0, .g = 250, .b = 154 }, try RGB.parse("medium spring green"));
    try testing.expectEqual(RGB{ .r = 34, .g = 139, .b = 34 }, try RGB.parse(" Forest Green "));

    // Invalid format
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb;"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:"));
    try testing.expectError(error.InvalidFormat, RGB.parse(":a/a/a"));
    try testing.expectError(error.InvalidFormat, RGB.parse("a/a/a"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:a/a/a/"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:00000///"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:000/"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgbi:a/a/a"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:0.5/0.0/1.0"));
    try testing.expectError(error.InvalidFormat, RGB.parse("rgb:not/hex/zz"));
    try testing.expectError(error.InvalidFormat, RGB.parse("#"));
    try testing.expectError(error.InvalidFormat, RGB.parse("#ff"));
    try testing.expectError(error.InvalidFormat, RGB.parse("#ffff"));
    try testing.expectError(error.InvalidFormat, RGB.parse("#fffff"));
    try testing.expectError(error.InvalidFormat, RGB.parse("#gggggg"));
}

test "DynamicPalette: init" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    try testing.expectEqual(default, p.current);
    try testing.expectEqual(default, p.original);
    try testing.expectEqual(@as(usize, 0), p.mask.count());
}

test "DynamicPalette: set" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    const new_color = RGB{ .r = 255, .g = 0, .b = 0 };

    p.set(0, new_color);
    try testing.expectEqual(new_color, p.current[0]);
    try testing.expect(p.mask.isSet(0));
    try testing.expectEqual(@as(usize, 1), p.mask.count());

    try testing.expectEqual(default[0], p.original[0]);
}

test "DynamicPalette: reset" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    const new_color = RGB{ .r = 255, .g = 0, .b = 0 };

    p.set(0, new_color);
    try testing.expect(p.mask.isSet(0));

    p.reset(0);
    try testing.expectEqual(default[0], p.current[0]);
    try testing.expect(!p.mask.isSet(0));
    try testing.expectEqual(@as(usize, 0), p.mask.count());
}

test "DynamicPalette: resetAll" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    const new_color = RGB{ .r = 255, .g = 0, .b = 0 };

    p.set(0, new_color);
    p.set(5, new_color);
    p.set(10, new_color);
    try testing.expectEqual(@as(usize, 3), p.mask.count());

    p.resetAll();
    try testing.expectEqual(default, p.current);
    try testing.expectEqual(default, p.original);
    try testing.expectEqual(@as(usize, 0), p.mask.count());
}

test "DynamicPalette: changeDefault with no changes" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    var new_palette = default;
    new_palette[0] = RGB{ .r = 100, .g = 100, .b = 100 };

    p.changeDefault(new_palette);
    try testing.expectEqual(new_palette, p.original);
    try testing.expectEqual(new_palette, p.current);
    try testing.expectEqual(@as(usize, 0), p.mask.count());
}

test "DynamicPalette: changeDefault preserves changes" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    const custom_color = RGB{ .r = 255, .g = 0, .b = 0 };

    p.set(5, custom_color);
    try testing.expect(p.mask.isSet(5));

    var new_palette = default;
    new_palette[0] = RGB{ .r = 100, .g = 100, .b = 100 };
    new_palette[5] = RGB{ .r = 50, .g = 50, .b = 50 };

    p.changeDefault(new_palette);

    try testing.expectEqual(new_palette, p.original);
    try testing.expectEqual(new_palette[0], p.current[0]);
    try testing.expectEqual(custom_color, p.current[5]);
    try testing.expect(p.mask.isSet(5));
    try testing.expectEqual(@as(usize, 1), p.mask.count());
}

test "DynamicPalette: changeDefault with multiple changes" {
    const testing = std.testing;

    var p: DynamicPalette = .init(default);
    const red = RGB{ .r = 255, .g = 0, .b = 0 };
    const green = RGB{ .r = 0, .g = 255, .b = 0 };
    const blue = RGB{ .r = 0, .g = 0, .b = 255 };

    p.set(1, red);
    p.set(2, green);
    p.set(3, blue);

    var new_palette = default;
    new_palette[0] = RGB{ .r = 50, .g = 50, .b = 50 };
    new_palette[1] = RGB{ .r = 60, .g = 60, .b = 60 };

    p.changeDefault(new_palette);

    try testing.expectEqual(new_palette[0], p.current[0]);
    try testing.expectEqual(red, p.current[1]);
    try testing.expectEqual(green, p.current[2]);
    try testing.expectEqual(blue, p.current[3]);
    try testing.expectEqual(@as(usize, 3), p.mask.count());
}

test "OkLab.fromRgb" {
    const testing = std.testing;
    const epsilon = 0.005;

    // White (255, 255, 255) -> l=1.0, a=0.0, b=0.0
    const white = OkLab.fromRgb(.{ .r = 255, .g = 255, .b = 255 });
    try testing.expectApproxEqAbs(@as(f32, 1.0), white.l, epsilon);
    try testing.expectApproxEqAbs(@as(f32, 0.0), white.a, epsilon);
    try testing.expectApproxEqAbs(@as(f32, 0.0), white.b, epsilon);

    // Black (0, 0, 0) -> l=0.0, a=0.0, b=0.0
    const black = OkLab.fromRgb(.{ .r = 0, .g = 0, .b = 0 });
    try testing.expectApproxEqAbs(@as(f32, 0.0), black.l, epsilon);
    try testing.expectApproxEqAbs(@as(f32, 0.0), black.a, epsilon);
    try testing.expectApproxEqAbs(@as(f32, 0.0), black.b, epsilon);
}

test "generate256Color: base16 preserved" {
    const testing = std.testing;

    const bg = RGB{ .r = 0, .g = 0, .b = 0 };
    const fg = RGB{ .r = 255, .g = 255, .b = 255 };
    const palette = generate256Color(default, .initEmpty(), bg, fg);

    // The first 16 colors (base16) must remain unchanged.
    for (0..16) |i| {
        try testing.expectEqual(default[i], palette[i]);
    }
}

test "generate256Color: cube corners match base colors" {
    const testing = std.testing;

    const bg = RGB{ .r = 0, .g = 0, .b = 0 };
    const fg = RGB{ .r = 255, .g = 255, .b = 255 };
    const palette = generate256Color(default, .initEmpty(), bg, fg);

    // Index 16 is cube (0,0,0) which should equal bg.
    try testing.expectEqual(bg, palette[16]);

    // Index 231 is cube (5,5,5) which should equal fg.
    try testing.expectEqual(fg, palette[231]);
}

test "generate256Color: grayscale ramp monotonic luminance" {
    const testing = std.testing;

    const bg = RGB{ .r = 0, .g = 0, .b = 0 };
    const fg = RGB{ .r = 255, .g = 255, .b = 255 };
    const palette = generate256Color(default, .initEmpty(), bg, fg);

    // The grayscale ramp (232–255) should have monotonically increasing
    // luminance from near-black to near-white.
    var prev_lum: f64 = 0.0;
    for (232..256) |i| {
        const lum = palette[i].luminance();
        try testing.expect(lum >= prev_lum);
        prev_lum = lum;
    }
}

test "generate256Color: skip mask preserves original colors" {
    const testing = std.testing;

    const bg = RGB{ .r = 0, .g = 0, .b = 0 };
    const fg = RGB{ .r = 255, .g = 255, .b = 255 };

    // Mark a few indices as skipped; they should keep their base value.
    var skip: PaletteMask = .initEmpty();
    skip.set(20);
    skip.set(100);
    skip.set(240);

    const palette = generate256Color(default, skip, bg, fg);
    try testing.expectEqual(default[20], palette[20]);
    try testing.expectEqual(default[100], palette[100]);
    try testing.expectEqual(default[240], palette[240]);

    // A non-skipped index in the cube should differ from the default.
    try testing.expect(!palette[21].eql(default[21]));
}

test "OkLab.toRgb" {
    const testing = std.testing;

    // Round-trip: RGB -> OkLab -> RGB should recover the original values.
    const cases = [_]RGB{
        .{ .r = 255, .g = 255, .b = 255 },
        .{ .r = 0, .g = 0, .b = 0 },
        .{ .r = 255, .g = 0, .b = 0 },
        .{ .r = 0, .g = 128, .b = 0 },
        .{ .r = 0, .g = 0, .b = 255 },
        .{ .r = 128, .g = 128, .b = 128 },
        .{ .r = 64, .g = 224, .b = 208 },
    };

    for (cases) |expected| {
        const lab = OkLab.fromRgb(expected);
        const actual = lab.toRgb();
        try testing.expectEqual(expected.r, actual.r);
        try testing.expectEqual(expected.g, actual.g);
        try testing.expectEqual(expected.b, actual.b);
    }
}
